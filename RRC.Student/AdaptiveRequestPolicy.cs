using RRC.Abstractions;

namespace RRC.Student;

// Adaptive timeout/retry policy.
//
// Key property of the system: a timeout does not cancel the server work. Every retry is an extra attempt
// that occupies the server, so retries are useful only against random slow outliers and harmful when
// requests are slow because of load. The policy therefore tries to keep timeouts rare unless the
// statistics clearly show that restarting a slow attempt is faster.
//
// 1. Recent attempt results are kept in a sliding window. A success gives an exact latency,
//    a timeout gives a censored value ("latency >= timeout"), because late responses are not reported.
// 2. The latency distribution is estimated with the Kaplan-Meier estimator, which handles censored values.
// 3. For every candidate timeout T the expected time to success with immediate restarts is computed:
//    E(T) = integral(0..T) S(t) dt / (1 - S(T)). Restarts are used only if they give a noticeable gain,
//    then the largest T whose E(T) is close to the minimum is chosen (fewer retries for the same latency).
//    Otherwise the timeout is set well above all observed latencies and only protects against outliers.
// 4. The restart formula assumes attempts are independent. The policy checks it twice:
//    - slow first attempts must not come in clusters of neighbouring requests (a cluster means
//      the slowness is caused by load or by the current period of time, not by bad luck);
//    - real retries must be fast as often as predicted, otherwise restarts are switched off.
// 5. A fast guard watches the timeout rate of first attempts. If it suddenly grows (degradation, overload),
//    the timeout is increased immediately, old statistics are dropped and retries get backoff with jitter,
//    so the policy does not create a retry storm.
// 6. The timeout grows immediately but decreases slowly: fast responses always arrive first,
//    so fresh statistics are optimistic while slow attempts are still in flight.
// 7. A request is never abandoned early: each next attempt gets a twice longer timeout.
public sealed class AdaptiveRequestPolicy : IRequestPolicy
{
    private const int WindowSize = 600;
    private const int MinSamples = 60;
    private const int RecomputeEvery = 20;
    private const int MaxAttempts = 8;
    private const double InitialTimeoutMs = 10_000;
    private const double MinTimeoutMs = 1;
    private const double MaxTimeoutMs = 60_000;

    // Restarts must reduce the expected latency at least by 10%, otherwise they are not worth the load.
    private const double RequiredGain = 0.9;
    private const double NearOptimum = 1.05;
    private const double MaxRestartShare = 0.15;
    private const double MaxClustering = 2;
    private const int MinClusterEvidence = 3;
    private const double TailMargin = 2;
    private const double InteriorMargin = 1.05;
    private const double MaxDecrease = 0.85;

    private const double GuardAlpha = 0.05;
    private const double GuardMinRate = 0.25;
    private const int GuardCooldown = 10;
    private const double GuardGrowth = 4;

    private const double RetryCheckAlpha = 0.1;
    private const int RetryCheckMinCount = 20;

    private readonly Sample[] window = new Sample[WindowSize];
    private int windowCount;
    private int windowNext;
    private int sinceRecompute;
    private int sinceGuard = GuardCooldown;

    private double timeoutMs = InitialTimeoutMs;
    private double expectedTimeoutRate;
    private double recentTimeoutRate;

    private bool restartsTrusted = true;
    private double retryFastRate = 1;
    private int retriesChecked;

    public TimeSpan GetTimeout(RequestContext context, Random random) =>
        TimeSpan.FromMilliseconds(TimeoutFor(context.Attempt));

    public RetryDecision OnResult(RequestObservation observation, Random random)
    {
        bool timedOut = observation.Outcome == RequestOutcome.Timeout;
        double elapsedMs = observation.Elapsed.TotalMilliseconds;
        Add(new Sample(observation.RequestId, observation.Attempt, elapsedMs, timedOut));

        if (observation.Attempt == 1)
        {
            UpdateGuard(timedOut);
        }
        else if (observation.Attempt == 2)
        {
            CheckRetryIndependence(!timedOut && elapsedMs <= timeoutMs);
        }

        if (++sinceRecompute >= RecomputeEvery && windowCount >= MinSamples)
        {
            sinceRecompute = 0;
            Recompute();
        }

        if (!timedOut || observation.Attempt >= MaxAttempts)
        {
            return RetryDecision.NoRetry();
        }

        // Isolated slow attempts are restarted immediately: waiting only adds latency.
        // When many attempts time out the server is likely overloaded, so retries are spread out.
        if (!IsOverloaded())
        {
            return RetryDecision.RetryAfter(TimeSpan.Zero);
        }

        double backoff = TimeoutFor(observation.Attempt) * random.NextDouble();
        return RetryDecision.RetryAfter(TimeSpan.FromMilliseconds(backoff));
    }

    private double TimeoutFor(int attempt)
    {
        double value = timeoutMs * Math.Pow(2, Math.Min(attempt - 1, 20));
        return Math.Clamp(value, MinTimeoutMs, MaxTimeoutMs);
    }

    private bool IsOverloaded() => recentTimeoutRate > Math.Max(GuardMinRate, 2 * expectedTimeoutRate + 0.15);

    private void UpdateGuard(bool timedOut)
    {
        recentTimeoutRate += GuardAlpha * ((timedOut ? 1 : 0) - recentTimeoutRate);
        sinceGuard++;
        if (!IsOverloaded() || sinceGuard < GuardCooldown)
        {
            return;
        }

        // Latency has shifted up: the old statistics describe another regime and would pull the timeout down.
        sinceGuard = 0;
        timeoutMs = Math.Min(timeoutMs * GuardGrowth, MaxTimeoutMs);
        expectedTimeoutRate = 0;
        windowCount = 0;
        windowNext = 0;
        sinceRecompute = 0;
    }

    private void CheckRetryIndependence(bool fast)
    {
        retryFastRate += RetryCheckAlpha * ((fast ? 1 : 0) - retryFastRate);
        retriesChecked++;
        // For independent attempts a retry finishes within the base timeout with probability 1 - S(T).
        if (retriesChecked >= RetryCheckMinCount && retryFastRate < 0.5 * (1 - expectedTimeoutRate))
        {
            restartsTrusted = false;
        }
    }

    private void Add(Sample sample)
    {
        window[windowNext] = sample;
        windowNext = (windowNext + 1) % WindowSize;
        windowCount = Math.Min(windowCount + 1, WindowSize);
    }

    private void Recompute()
    {
        var samples = new Sample[windowCount];
        Array.Copy(window, samples, windowCount);
        // Successes go before timeouts with the same value, as Kaplan-Meier requires.
        Array.Sort(samples, (a, b) => a.Value != b.Value ? a.Value.CompareTo(b.Value) : a.Censored.CompareTo(b.Censored));

        var candidates = new List<Candidate>();
        double survival = 1;
        double area = 0;
        double previous = 0;
        int atRisk = samples.Length;
        foreach (var sample in samples)
        {
            area += survival * (sample.Value - previous);
            previous = sample.Value;
            if (!sample.Censored)
            {
                survival *= 1 - 1.0 / atRisk;
                if (survival <= MaxRestartShare)
                {
                    candidates.Add(new Candidate(sample.Value, area / (1 - survival), survival));
                }
            }
            atRisk--;
        }

        if (candidates.Count == 0)
        {
            // Too many attempts time out: the timeout is far below the real latency.
            timeoutMs = Math.Min(timeoutMs * 2, MaxTimeoutMs);
            expectedTimeoutRate = 0;
            return;
        }

        // Without restarts the mean latency is at least the area under S up to the largest observed value.
        double noRestartLatency = area;
        var last = candidates[^1];
        double best = candidates.Min(candidate => candidate.ExpectedLatency);
        // Among nearly optimal timeouts the largest one gives fewer retries and less extra load.
        var chosen = candidates.Last(candidate => candidate.ExpectedLatency <= best * NearOptimum);

        if (restartsTrusted && best < RequiredGain * noRestartLatency && !IsClustered(samples, chosen.Value))
        {
            SetTimeout(chosen.Value * InteriorMargin);
            expectedTimeoutRate = chosen.Survival;
        }
        else
        {
            // Restarting does not help: keep the timeout above everything observed so far,
            // so it only catches rare outliers and the tail stays visible.
            SetTimeout(last.Value * TailMargin);
            expectedTimeoutRate = last.Survival;
        }
    }

    private void SetTimeout(double value) =>
        timeoutMs = Math.Clamp(Math.Max(value, timeoutMs * MaxDecrease), MinTimeoutMs, MaxTimeoutMs);

    // Compares the chance that the next request is slow after a slow one with the overall chance.
    // For independent latencies they are equal; load spikes and degradations make slow requests neighbours.
    private static bool IsClustered(Sample[] samples, double threshold)
    {
        var first = samples.Where(sample => sample.Attempt == 1).OrderBy(sample => sample.RequestId).ToArray();
        int slow = 0, pairs = 0, slowAfterSlow = 0;
        for (int i = 0; i < first.Length; i++)
        {
            bool isSlow = first[i].Censored || first[i].Value > threshold;
            if (isSlow)
            {
                slow++;
            }
            // Only really consecutive requests are compared: slow answers arrive late, so the window
            // also holds old slow requests whose neighbours have already left it.
            if (i == 0 || first[i].RequestId != first[i - 1].RequestId + 1)
            {
                continue;
            }
            if (!(first[i - 1].Censored || first[i - 1].Value > threshold))
            {
                continue;
            }
            pairs++;
            if (isSlow)
            {
                slowAfterSlow++;
            }
        }
        // A few neighbouring slow requests can happen by chance, a cluster needs several of them.
        if (slowAfterSlow < MinClusterEvidence)
        {
            return false;
        }
        double overall = (double)slow / first.Length;
        double afterSlow = (double)slowAfterSlow / pairs;
        return afterSlow > MaxClustering * overall + 0.1;
    }

    private readonly struct Sample
    {
        public long RequestId { get; }
        public int Attempt { get; }
        public double Value { get; }
        public bool Censored { get; }

        public Sample(
            long requestId,
            int attempt,
            double value,
            bool censored)
        {
            RequestId = requestId;
            Attempt = attempt;
            Value = value;
            Censored = censored;
        }
    }

    private readonly struct Candidate
    {
        public double Value { get; }
        public double ExpectedLatency { get; }
        public double Survival { get; }

        public Candidate(
            double value,
            double expectedLatency,
            double survival)
        {
            Value = value;
            ExpectedLatency = expectedLatency;
            Survival = survival;
        }
    }
}
