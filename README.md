# vectorbench

Sorry, you need to install maven, it uses JMH.

## how to run all benchmarks

```console
$ mvn verify
$ java -jar target/vectorbench.jar
```

## listing benchmarks available

```console
$ java -jar target/vectorbench.jar -l
```

## run specific benchmark

```console
$ java -jar target/vectorbench.jar BinaryDotProduct
```
