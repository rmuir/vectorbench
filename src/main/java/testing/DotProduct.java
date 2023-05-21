package testing;

import org.openjdk.jmh.annotations.*;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;


@State(Scope.Thread)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class DotProduct {

  private float[] a;
  private float[] b;

  //@Param({"1", "4", "6", "8", "13", "16", "25", "32", "64", "100", "128", "207", "256", "300", "512", "702", "1024"})
  //@Param({"1", "4", "6", "8", "13", "16", "25", "32", "64", "100" })
  @Param({"1024"})
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
    float s0 = 0f;
    float s1 = 0f;
    float s2 = 0f;
    float s3 = 0f;
    float s4 = 0f;
    float s5 = 0f;
    float s6 = 0f;
    float s7 = 0f;
    for (; i + 7 < a.length; i += 8) {
      s0 += b[i] * a[i];
      s1 += b[i + 1] * a[i + 1];
      s2 += b[i + 2] * a[i + 2];
      s3 += b[i + 3] * a[i + 3];
      s4 += b[i + 4] * a[i + 4];
      s5 += b[i + 5] * a[i + 5];
      s6 += b[i + 6] * a[i + 6];
      s7 += b[i + 7] * a[i + 7];
    }
    res += s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    return res;
  }
}
