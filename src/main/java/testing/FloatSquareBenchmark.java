package testing;

import org.openjdk.jmh.annotations.*;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 3)
@Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
public class FloatSquareBenchmark {

  private float[] a;
  private float[] b;

  @Param({"1", "4", "6", "8", "13", "16", "25", "32", "64", "100", "128", "207", "256", "300", "512", "702", "1024"})
  //@Param({"1", "4", "6", "8", "13", "16", "25", "32", "64", "100" })
  //@Param({"702", "1024"})
  //@Param({"16", "32", "64"})
  //@Param({"1024"})
  int size;

  @Setup(Level.Trial)
  public void init() {
    a = new float[size];
    b = new float[size];
    for (int i = 0; i < size; ++i) {
      a[i] = ThreadLocalRandom.current().nextFloat();
      b[i] = ThreadLocalRandom.current().nextFloat();
    }
    // order of ops may change, but try to detect broken shit
    if (Math.abs(squareOld() - squareNew()) > 0.001f) {
      throw new RuntimeException("probably wrong");
    }
  }

  static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  @Benchmark
  public float squareNew() {
    if (a.length != b.length) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length + "!=" + b.length);
    }
    int i = 0;
    float res = 0;
    // if the array size is large (> 2x platform vector size), its worth the overhead to vectorize
    if (a.length > 2 * SPECIES.length()) {
      // vector loop is unrolled 4x (4 accumulators in parallel)
      FloatVector acc1 = FloatVector.zero(SPECIES);
      FloatVector acc2 = FloatVector.zero(SPECIES);
      FloatVector acc3 = FloatVector.zero(SPECIES);
      FloatVector acc4 = FloatVector.zero(SPECIES);
      int upperBound = SPECIES.loopBound(a.length - 3*SPECIES.length());
      for (; i < upperBound; i += 4 * SPECIES.length()) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        FloatVector diff1 = va.sub(vb);
        acc1 = acc1.add(diff1.mul(diff1));
        FloatVector vc = FloatVector.fromArray(SPECIES, a, i + SPECIES.length());
        FloatVector vd = FloatVector.fromArray(SPECIES, b, i + SPECIES.length());
        FloatVector diff2 = vc.sub(vd);
        acc2 = acc2.add(diff2.mul(diff2));
        FloatVector ve = FloatVector.fromArray(SPECIES, a, i + 2*SPECIES.length());
        FloatVector vf = FloatVector.fromArray(SPECIES, b, i + 2*SPECIES.length());
        FloatVector diff3 = ve.sub(vf);
        acc3 = acc3.add(diff3.mul(diff3));
        FloatVector vg = FloatVector.fromArray(SPECIES, a, i + 3*SPECIES.length());
        FloatVector vh = FloatVector.fromArray(SPECIES, b, i + 3*SPECIES.length());
        FloatVector diff4 = vg.sub(vh);
        acc4 = acc4.add(diff4.mul(diff4));
      }
      // vector tail: less scalar computations for unaligned sizes, esp with big vector sizes
      upperBound = SPECIES.loopBound(a.length);
      for (; i < upperBound; i += SPECIES.length()) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        FloatVector diff = va.sub(vb);
        acc1 = acc1.add(diff.mul(diff));
      }
      // reduce
      FloatVector res1 = acc1.add(acc2);
      FloatVector res2 = acc3.add(acc4);
      res += res1.add(res2).reduceLanes(VectorOperators.ADD);
    }

    for (; i < a.length; i++) {
      float diff = a[i] - b[i];
      res += diff * diff;
    }
    return res;
  }

  /**
   * Returns the sum of squared differences of the two vectors.
   *
   * @throws IllegalArgumentException if the vectors' dimensions differ.
   */
  @Benchmark
  public float squareOld() {
    if (a.length != b.length) {
      throw new IllegalArgumentException(
          "vector dimensions differ: " + a.length + "!=" + b.length);
    }
    float squareSum = 0.0f;
    int dim = a.length;
    int i;
    for (i = 0; i + 8 <= dim; i += 8) {
      squareSum += squareDistanceUnrolled(a, b, i);
    }
    for (; i < dim; i++) {
      float diff = a[i] - b[i];
      squareSum += diff * diff;
    }
    return squareSum;
  }

  private static float squareDistanceUnrolled(float[] v1, float[] v2, int index) {
    float diff0 = v1[index + 0] - v2[index + 0];
    float diff1 = v1[index + 1] - v2[index + 1];
    float diff2 = v1[index + 2] - v2[index + 2];
    float diff3 = v1[index + 3] - v2[index + 3];
    float diff4 = v1[index + 4] - v2[index + 4];
    float diff5 = v1[index + 5] - v2[index + 5];
    float diff6 = v1[index + 6] - v2[index + 6];
    float diff7 = v1[index + 7] - v2[index + 7];
    return diff0 * diff0
        + diff1 * diff1
        + diff2 * diff2
        + diff3 * diff3
        + diff4 * diff4
        + diff5 * diff5
        + diff6 * diff6
        + diff7 * diff7;
  }
}
