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
public class FloatCosineBenchmark {

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
    if (Math.abs(cosineOld() - cosineNew()) > 0.001f) {
      throw new RuntimeException("probably wrong");
    }
  }

  static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  @Benchmark
  public float cosineNew() {
    if (a.length != b.length) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length + "!=" + b.length);
    }
    int i = 0;
    float sum = 0;
    float norm1 = 0;
    float norm2 = 0;
    // if the array size is large (> 2x platform vector size), its worth the overhead to vectorize
    if (a.length > 2 * SPECIES.length()) {
      // vector loop is unrolled 4x (4 accumulators in parallel)
      FloatVector sum1 = FloatVector.zero(SPECIES);
      FloatVector sum2 = FloatVector.zero(SPECIES);
      FloatVector sum3 = FloatVector.zero(SPECIES);
      FloatVector sum4 = FloatVector.zero(SPECIES);
      FloatVector norm1_1 = FloatVector.zero(SPECIES);
      FloatVector norm1_2 = FloatVector.zero(SPECIES);
      FloatVector norm1_3 = FloatVector.zero(SPECIES);
      FloatVector norm1_4 = FloatVector.zero(SPECIES);
      FloatVector norm2_1 = FloatVector.zero(SPECIES);
      FloatVector norm2_2 = FloatVector.zero(SPECIES);
      FloatVector norm2_3 = FloatVector.zero(SPECIES);
      FloatVector norm2_4 = FloatVector.zero(SPECIES);
      int upperBound = SPECIES.loopBound(a.length - 3*SPECIES.length());
      for (; i < upperBound; i += 4 * SPECIES.length()) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        sum1 = sum1.add(va.mul(vb));
        norm1_1 = norm1_1.add(va.mul(va));
        norm2_1 = norm2_1.add(vb.mul(vb));
        FloatVector vc = FloatVector.fromArray(SPECIES, a, i + SPECIES.length());
        FloatVector vd = FloatVector.fromArray(SPECIES, b, i + SPECIES.length());
        sum2 = sum2.add(vc.mul(vd));
        norm1_2 = norm1_2.add(vc.mul(vc));
        norm2_2 = norm2_2.add(vd.mul(vd));
        FloatVector ve = FloatVector.fromArray(SPECIES, a, i + 2*SPECIES.length());
        FloatVector vf = FloatVector.fromArray(SPECIES, b, i + 2*SPECIES.length());
        sum3 = sum3.add(ve.mul(vf));
        norm1_3 = norm1_3.add(ve.mul(ve));
        norm2_3 = norm2_3.add(vf.mul(vf));
        FloatVector vg = FloatVector.fromArray(SPECIES, a, i + 3*SPECIES.length());
        FloatVector vh = FloatVector.fromArray(SPECIES, b, i + 3*SPECIES.length());
        sum4 = sum4.add(vg.mul(vh));
        norm1_4 = norm1_4.add(vg.mul(vg));
        norm2_4 = norm2_4.add(vh.mul(vh));
      }
      // vector tail: less scalar computations for unaligned sizes, esp with big vector sizes
      upperBound = SPECIES.loopBound(a.length);
      for (; i < upperBound; i += SPECIES.length()) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        sum1 = sum1.add(va.mul(vb));
        norm1_1 = norm1_1.add(va.mul(va));
        norm2_1 = norm2_1.add(vb.mul(vb));
      }
      // reduce
      FloatVector sumres1 = sum1.add(sum2);
      FloatVector sumres2 = sum3.add(sum4);
      FloatVector norm1res1 = norm1_1.add(norm1_2);
      FloatVector norm1res2 = norm1_3.add(norm1_4);
      FloatVector norm2res1 = norm2_1.add(norm2_2);
      FloatVector norm2res2 = norm2_3.add(norm2_4);
      sum += sumres1.add(sumres2).reduceLanes(VectorOperators.ADD);
      norm1 += norm1res1.add(norm1res2).reduceLanes(VectorOperators.ADD);
      norm2 += norm2res1.add(norm2res2).reduceLanes(VectorOperators.ADD);
    }

    for (; i < a.length; i++) {
      float elem1 = a[i];
      float elem2 = b[i];
      sum += elem1 * elem2;
      norm1 += elem1 * elem1;
      norm2 += elem2 * elem2;
    }
    return (float) (sum / Math.sqrt(norm1 * norm2));
  }

  @Benchmark
  public float cosineOld() {
    if (a.length != b.length) {
      throw new IllegalArgumentException(
          "vector dimensions differ: " + a.length + "!=" + b.length);
    }

    float sum = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    int dim = a.length;

    for (int i = 0; i < dim; i++) {
      float elem1 = a[i];
      float elem2 = b[i];
      sum += elem1 * elem2;
      norm1 += elem1 * elem1;
      norm2 += elem2 * elem2;
    }
    return (float) (sum / Math.sqrt(norm1 * norm2));
  }
}
