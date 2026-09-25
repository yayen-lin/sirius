## Run Eval

- scaling curve 1 on as data size grow
- scaling curve 2 on self join
- scaling curve 3 on as dimensionality grows
- scaling curve 4 on as probe size grows, against a fixed 10M corpus (reuses scaling curve 2's prep)

```bash

# prep (to generate dim and slice)
./bench/vector/scaling-curve-2/prep.sh; ./bench/vector/scaling-curve-3/prep.sh; 

# run
./bench/vector/scaling-curve-1/run.sh; ./bench/vector/scaling-curve-2/run.sh; ./bench/vector/scaling-curve-3/run.sh; ./bench/vector/scaling-curve-4/run.sh
```