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
public class DotProductBenchmark {

  private float[] a;
  private float[] b;

  @Param({"1", "4", "6", "8", "13", "16", "25", "32", "64", "100", "128", "207", "256", "300", "512", "702", "1024"})
  //@Param({"1", "4", "6", "8", "13", "16", "25", "32", "64", "100" })
  //@Param({"702", "1024"})
  //@Param({"16", "32", "64"})
  int size;

  @Setup(Level.Trial)
  public void init() {
    a = new float[size];
    b = new float[size];
    for (int i = 0; i < size; ++i) {
      a[i] = ThreadLocalRandom.current().nextFloat();
      b[i] = ThreadLocalRandom.current().nextFloat();
    }
  }

  static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  @Benchmark
  public float dotProductNewNew() {
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
        acc1 = acc1.add(va.mul(vb));
        FloatVector vc = FloatVector.fromArray(SPECIES, a, i + SPECIES.length());
        FloatVector vd = FloatVector.fromArray(SPECIES, b, i + SPECIES.length());
        acc2 = acc2.add(vc.mul(vd));
        FloatVector ve = FloatVector.fromArray(SPECIES, a, i + 2*SPECIES.length());
        FloatVector vf = FloatVector.fromArray(SPECIES, b, i + 2*SPECIES.length());
        acc3 = acc3.add(ve.mul(vf));
        FloatVector vg = FloatVector.fromArray(SPECIES, a, i + 3*SPECIES.length());
        FloatVector vh = FloatVector.fromArray(SPECIES, b, i + 3*SPECIES.length());
        acc4 = acc4.add(vg.mul(vh));
      }
      // vector tail: less scalar computations for unaligned sizes, esp with big vector sizes
      upperBound = SPECIES.loopBound(a.length);
      for (; i < upperBound; i += SPECIES.length()) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        acc1 = acc1.add(va.mul(vb));
      }
      // reduce
      FloatVector res1 = acc1.add(acc2);
      FloatVector res2 = acc3.add(acc4);
      res += res1.add(res2).reduceLanes(VectorOperators.ADD);
    }

    for (; i < a.length; i++) {
      res += b[i] * a[i];
    }
    return res;
  }

  @Benchmark
  public float dotProductNew() {
    if (a.length != b.length) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length + "!=" + b.length);
    }
    int i = 0;
    float res = 0;
    // if the array size is large (2x platform vector size), its worth the overhead to vectorize
    // vector loop is unrolled a single time (2 accumulators in parallel)
    if (a.length >= 2 * SPECIES.length()) {
      FloatVector acc1 = FloatVector.zero(SPECIES);
      FloatVector acc2 = FloatVector.zero(SPECIES);
      int upperBound = SPECIES.loopBound(a.length - SPECIES.length());
      for (; i < upperBound; i += 2 * SPECIES.length()) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        acc1 = acc1.add(va.mul(vb));
        FloatVector vc = FloatVector.fromArray(SPECIES, a, i + SPECIES.length());
        FloatVector vd = FloatVector.fromArray(SPECIES, b, i + SPECIES.length());
        acc2 = acc2.add(vc.mul(vd));
      }
      res += acc1.reduceLanes(VectorOperators.ADD) + acc2.reduceLanes(VectorOperators.ADD);
    }
    for (; i < a.length; i++) {
      res += b[i] * a[i];
    }
    return res;
  }

  @Benchmark
  public float dotProductOld() {
    if (a.length != b.length) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length + "!=" + b.length);
    }
    float res = 0f;
    /*
     * If length of vector is larger than 8, we use unrolled dot product to accelerate the
     * calculation.
     */
    int i;
    for (i = 0; i < a.length % 8; i++) {
      res += b[i] * a[i];
    }
    if (a.length < 8) {
      return res;
    }
    for (; i + 31 < a.length; i += 32) {
      res +=
          b[i + 0] * a[i + 0]
              + b[i + 1] * a[i + 1]
              + b[i + 2] * a[i + 2]
              + b[i + 3] * a[i + 3]
              + b[i + 4] * a[i + 4]
              + b[i + 5] * a[i + 5]
              + b[i + 6] * a[i + 6]
              + b[i + 7] * a[i + 7];
      res +=
          b[i + 8] * a[i + 8]
              + b[i + 9] * a[i + 9]
              + b[i + 10] * a[i + 10]
              + b[i + 11] * a[i + 11]
              + b[i + 12] * a[i + 12]
              + b[i + 13] * a[i + 13]
              + b[i + 14] * a[i + 14]
              + b[i + 15] * a[i + 15];
      res +=
          b[i + 16] * a[i + 16]
              + b[i + 17] * a[i + 17]
              + b[i + 18] * a[i + 18]
              + b[i + 19] * a[i + 19]
              + b[i + 20] * a[i + 20]
              + b[i + 21] * a[i + 21]
              + b[i + 22] * a[i + 22]
              + b[i + 23] * a[i + 23];
      res +=
          b[i + 24] * a[i + 24]
              + b[i + 25] * a[i + 25]
              + b[i + 26] * a[i + 26]
              + b[i + 27] * a[i + 27]
              + b[i + 28] * a[i + 28]
              + b[i + 29] * a[i + 29]
              + b[i + 30] * a[i + 30]
              + b[i + 31] * a[i + 31];
    }
    for (; i + 7 < a.length; i += 8) {
      res +=
          b[i + 0] * a[i + 0]
              + b[i + 1] * a[i + 1]
              + b[i + 2] * a[i + 2]
              + b[i + 3] * a[i + 3]
              + b[i + 4] * a[i + 4]
              + b[i + 5] * a[i + 5]
              + b[i + 6] * a[i + 6]
              + b[i + 7] * a[i + 7];
    }
    return res;
  }
}
