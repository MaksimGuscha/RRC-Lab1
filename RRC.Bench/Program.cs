using System.Globalization;
using RRC.Abstractions;
using RRC.Simulation;
using RRC.Simulation.Public;
using RRC.Student;

// Stress bench: runs several policies on the public scenario and on harder hand-made scenarios
// (heavy tails, bursts, degradation, near/over capacity) for several seeds and prints mean metrics.
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

int seeds = args.Length > 0 ? int.Parse(args[0]) : 5;

var policies = new (string Name, Func<IRequestPolicy> Create)[]
{
    ("Sample", () => new SampleRequestPolicy()),
    ("Fixed+3retries", () => new FixedRetryPolicy()),
    ("NoRetry(10s)", () => new NoRetryPolicy(10)),
    ("NoRetry(60s)", () => new NoRetryPolicy(60)),
    ("Adaptive", () => new AdaptiveRequestPolicy()),
};

var scenarios = new Func<SimulationScenario>[]
{
    PublicScenario.Create,
    () => Make("Bimodal tail", new ConstantRateArrivalModel(100),
        new MixtureLatency(Uniform(50, 150), Uniform(1000, 3000), 0.05), 20),
    () => Make("Bimodal tail, cap 40", new ConstantRateArrivalModel(100),
        new MixtureLatency(Uniform(50, 150), Uniform(1000, 3000), 0.05), 40),
    () => Make("Rare huge outliers", new ConstantRateArrivalModel(100),
        new MixtureLatency(Uniform(20, 60), Uniform(5000, 8000), 0.01), 20),
    () => Make("Wave latency", new ConstantRateArrivalModel(100),
        new WaveLatency(Uniform(50, 150), 10, 0.8), 20),
    () => Make("Lognormal", new ConstantRateArrivalModel(100),
        new LogNormalLatency(80, 1.0), 30),
    () => Make("Poisson bursts", new BurstyArrival(100, 400, 5, 1),
        Uniform(50, 150), 20),
    () => Make("Degradation 10-15s x8", new ConstantRateArrivalModel(100),
        new DegradationLatency(Uniform(50, 150), 10, 15, 8), 20),
    () => Make("Near capacity", new ConstantRateArrivalModel(180),
        Uniform(50, 150), 20),
    () => Make("Slow service", new ConstantRateArrivalModel(20),
        Uniform(500, 1500), 20),
    () => Make("Fast service", new ConstantRateArrivalModel(1000),
        Uniform(2, 10), 10),
    () => Make("Overload", new ConstantRateArrivalModel(220),
        Uniform(50, 150), 20),
};

Console.WriteLine($"{"Scenario",-24}{"Policy",-16}{"Success",9}{"Mean ms",10}{"p95 ms",10}{"Att/req",9}");
foreach (var createScenario in scenarios)
{
    foreach (var (name, create) in policies)
    {
        double success = 0, mean = 0, p95 = 0, attempts = 0;
        for (int seed = 1; seed <= seeds; seed++)
        {
            var result = new SimulationRunner().Run(createScenario(), create(), new SimulationOptions(seed));
            success += result.Metrics.SuccessRate;
            mean += result.Metrics.MeanLatency?.TotalMilliseconds ?? double.NaN;
            p95 += result.Metrics.P95Latency?.TotalMilliseconds ?? double.NaN;
            attempts += result.Metrics.AttemptsPerRequest;
        }
        Console.WriteLine($"{createScenario().Name,-24}{name,-16}{success / seeds,9:P1}{mean / seeds,10:F1}{p95 / seeds,10:F1}{attempts / seeds,9:F3}");
    }
    Console.WriteLine();
}

static SimulationScenario Make(string name, IArrivalModel arrivals, ILatencyModel latency, int capacity) =>
    new(name, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(40), arrivals, latency, new CapacityLoadModel(capacity), name);

static ILatencyModel Uniform(double min, double max) =>
    new SimpleRandomLatencyModel(TimeSpan.FromMilliseconds(min), TimeSpan.FromMilliseconds(max));

// A typical "simple" baseline: fixed timeout, up to 3 retries with exponential backoff.
sealed class FixedRetryPolicy : IRequestPolicy
{
    public TimeSpan GetTimeout(RequestContext context, Random random) => TimeSpan.FromMilliseconds(200);

    public RetryDecision OnResult(RequestObservation observation, Random random) =>
        observation.Outcome == RequestOutcome.Timeout && observation.Attempt <= 3
            ? RetryDecision.RetryAfter(TimeSpan.FromMilliseconds(50 * Math.Pow(2, observation.Attempt - 1)))
            : RetryDecision.NoRetry();
}

// Waits long and never retries: no extra load, but no protection against slow outliers.
sealed class NoRetryPolicy(double seconds) : IRequestPolicy
{
    public TimeSpan GetTimeout(RequestContext context, Random random) => TimeSpan.FromSeconds(seconds);
    public RetryDecision OnResult(RequestObservation observation, Random random) => RetryDecision.NoRetry();
}

sealed class MixtureLatency(ILatencyModel main, ILatencyModel slow, double slowShare) : ILatencyModel
{
    public TimeSpan GetBaseLatency(TimeSpan currentTime, Random random) =>
        random.NextDouble() < slowShare ? slow.GetBaseLatency(currentTime, random) : main.GetBaseLatency(currentTime, random);
}

sealed class LogNormalLatency(double medianMs, double sigma) : ILatencyModel
{
    public TimeSpan GetBaseLatency(TimeSpan currentTime, Random random)
    {
        double u1 = 1 - random.NextDouble(), u2 = random.NextDouble();
        double normal = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        return TimeSpan.FromMilliseconds(Math.Min(medianMs * Math.Exp(sigma * normal), 30_000));
    }
}

sealed class DegradationLatency(ILatencyModel inner, double fromS, double toS, double factor) : ILatencyModel
{
    public TimeSpan GetBaseLatency(TimeSpan currentTime, Random random)
    {
        var value = inner.GetBaseLatency(currentTime, random);
        double t = currentTime.TotalSeconds;
        return t >= fromS && t < toS ? value * factor : value;
    }
}

sealed class WaveLatency(ILatencyModel inner, double periodS, double amplitude) : ILatencyModel
{
    public TimeSpan GetBaseLatency(TimeSpan currentTime, Random random) =>
        inner.GetBaseLatency(currentTime, random) * (1 + amplitude * Math.Sin(2 * Math.PI * currentTime.TotalSeconds / periodS));
}

// Poisson arrivals; every periodS seconds the rate jumps to burstRate for burstS seconds.
sealed class BurstyArrival(double rate, double burstRate, double periodS, double burstS) : IArrivalModel
{
    public TimeSpan GetNextDelay(TimeSpan currentTime, Random random)
    {
        double current = currentTime.TotalSeconds % periodS < burstS ? burstRate : rate;
        double seconds = -Math.Log(1 - random.NextDouble()) / current;
        return TimeSpan.FromTicks(Math.Max(1, (long)(seconds * TimeSpan.TicksPerSecond)));
    }
}
