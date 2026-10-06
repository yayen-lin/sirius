threshold eval

1. varying corpus size
2. self join with data size
3. varying dimension
4. varying probe size

As the probe side grows from 1 to 10k rows, Sirius's throughput grows about 1,500x, while DuckDB's grows about 4x.

```bash
# threshold: varying corpus size
engine       corpus reps        rows   dim  probe_rows  corpus_rows       pairs    dist_ops      min_ms     mean_ms      max_ms       ms_per_op billion_dist_ops_per_sec
duckdb     bigann1m   10    12404959   128       10000      1000000       1e+10    1.28e+12    158208.0    159060.5    160625.0    0.0000001243                     8.05
duckdb    bigann10m   10    52974965   128       10000     10000000       1e+11    1.28e+13    401044.0    402052.6    404934.0    0.0000000314                    31.84
duckdb   bigann100m   10     TIMEOUT   128       10000    100000000       1e+12    1.28e+14     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
sirius     bigann1m   10    12404959   128       10000      1000000       1e+10    1.28e+12       524.0       537.3       603.0    0.0000000004                  2382.28
sirius    bigann10m   10    52974965   128       10000     10000000       1e+11    1.28e+13      4183.0      4313.6      4583.0    0.0000000003                  2967.36
sirius   bigann100m   10   520018995   128       10000    100000000       1e+12    1.28e+14     51702.0     53131.8     54807.0    0.0000000004                  2409.10

# threshold: self join varying data size
engine         size reps        rows   dim  probe_rows  corpus_rows       pairs    dist_ops      min_ms     mean_ms      max_ms       ms_per_op billion_dist_ops_per_sec
duckdb           1k   10        1374   128        1000         1000       1e+06    1.28e+08       122.0       124.6       127.0    0.0000009734                     1.03
duckdb          10k   10       77468   128       10000        10000       1e+08    1.28e+10     12105.0     12222.8     12437.0    0.0000009549                     1.05
duckdb         100k   10     5069692   128      100000       100000       1e+10    1.28e+12   1200359.0   1210570.8   1224095.0    0.0000009458                     1.06
duckdb           1m   10     TIMEOUT   128     1000000      1000000       1e+12    1.28e+14     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
duckdb          10m   10     TIMEOUT   128    10000000     10000000       1e+14    1.28e+16     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
sirius           1k   10        1374   128        1000         1000       1e+06    1.28e+08        10.0        11.5        21.0    0.0000000898                    11.13
sirius          10k   10       77468   128       10000        10000       1e+08    1.28e+10        16.0        19.4        35.0    0.0000000015                   659.79
sirius         100k   10     5069692   128      100000       100000       1e+10    1.28e+12       398.0       405.2       447.0    0.0000000003                  3158.93
sirius           1m   10   524865066   128     1000000      1000000       1e+12    1.28e+14     37297.0     38660.8     39020.0    0.0000000003                  3310.85
sirius          10m   10 56234686976   128    10000000     10000000       1e+14    1.28e+16   3896932.0   3901356.5   3910846.0    0.0000000003                  3280.91

# threshold: varying dimension
engine          dim reps        rows        probe_rows  corpus_rows       pairs    dist_ops      min_ms     mean_ms      max_ms       ms_per_op billion_dist_ops_per_sec
duckdb          128   10   230428916              1000      1000000       1e+09    1.28e+11     16863.0     16971.9     17169.0    0.0000001326                     7.54
duckdb          256   10    13597287              1000      1000000       1e+09    2.56e+11     35429.0     35646.3     35908.0    0.0000001392                     7.18
duckdb          512   10     1058406              1000      1000000       1e+09    5.12e+11     73322.0     73546.7     73937.0    0.0000001436                     6.96
duckdb          768   10      211676              1000      1000000       1e+09    7.68e+11    112895.0    113663.0    114874.0    0.0000001480                     6.76
duckdb          960   10       81443              1000      1000000       1e+09     9.6e+11    141948.0    142591.1    143435.0    0.0000001485                     6.73
sirius          128   10   230428935              1000      1000000       1e+09    1.28e+11       256.0       277.3       312.0    0.0000000022                   461.59
sirius          256   10    13597294              1000      1000000       1e+09    2.56e+11       289.0       301.1       312.0    0.0000000012                   850.22
sirius          512   10     1058404              1000      1000000       1e+09    5.12e+11       487.0       571.1       614.0    0.0000000011                   896.52
sirius          768   10      211676              1000      1000000       1e+09    7.68e+11       707.0       810.6       904.0    0.0000000011                   947.45
sirius          960   10       81443              1000      1000000       1e+09     9.6e+11       803.0       858.5       905.0    0.0000000009                  1118.23


# threshold: varying probe size
#   - what should i do with join vs search?
engine        probe reps        rows   dim  probe_rows  corpus_rows       pairs    dist_ops      min_ms     mean_ms      max_ms       ms_per_op billion_dist_ops_per_sec
duckdb            1   10          85   128           1     10000000       1e+07    1.28e+09       151.0       162.4       180.0    0.0000001269                     7.88
duckdb           10   10         701   128          10     10000000       1e+08    1.28e+10       524.0       536.5       563.0    0.0000000419                    23.86
duckdb          100   10       65161   128         100     10000000       1e+09    1.28e+11      4142.0      4169.5      4194.0    0.0000000326                    30.70
duckdb           1k   10     1674893   128        1000     10000000       1e+10    1.28e+12     40166.0     40225.3     40274.0    0.0000000314                    31.82
duckdb          10k   10    58920227   128       10000     10000000       1e+11    1.28e+13    400533.0    401827.2    402429.0    0.0000000314                    31.85
duckdb         100k   10     TIMEOUT   128      100000     10000000       1e+12    1.28e+14     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
duckdb           1m   10     TIMEOUT   128     1000000     10000000       1e+13    1.28e+15     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
duckdb          10m   10     TIMEOUT   128    10000000     10000000       1e+14    1.28e+16     TIMEOUT     TIMEOUT     TIMEOUT               -                        -
sirius            1   10          85   128           1     10000000       1e+07    1.28e+09       603.0       635.8       739.0    0.0000004967                     2.01
sirius           10   10         701   128          10     10000000       1e+08    1.28e+10       749.0       766.5       775.0    0.0000000599                    16.70
sirius          100   10       65161   128         100     10000000       1e+09    1.28e+11       654.0       749.2       846.0    0.0000000059                   170.85
sirius           1k   10     1674893   128        1000     10000000       1e+10    1.28e+12      1002.0      1022.3      1039.0    0.0000000008                  1252.08
sirius          10k   10    58920227   128       10000     10000000       1e+11    1.28e+13      4165.0      4210.6      4250.0    0.0000000003                  3039.95
sirius         100k   10   510250913   128      100000     10000000       1e+12    1.28e+14     37395.0     38711.9     39059.0    0.0000000003                  3306.48
sirius           1m   10  5410900720   128     1000000     10000000       1e+13    1.28e+15    381177.0    383472.2    383940.0    0.0000000003                  3337.92
sirius          10m   10 56234686976   128    10000000     10000000       1e+14    1.28e+16   3883319.0   3892422.6   3901156.0    0.0000000003                  3288.44


```
