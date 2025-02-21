mkdir log

for expname in EXP_BENCH
do
for benchdataset in BENCH_B BENCH_C
do
for label in r1
do
for feat in Alpha360
do
for ht in 1 3 5 7 10 15 20
do
market=csi300
dt=$(date +"%Y%m%d_%H%M%S")
python ./src/bench.py --expname $expname_$dt --benchdataset $benchdataset --market $market  --topk 5 --ndrop 1 --label $label --ht $ht --feat $feat --retrain --savecsv > ./log/$dt.log
dt=$(date +"%Y%m%d_%H%M%S")
python ./src/bench.py --expname $expname_$dt --benchdataset $benchdataset --market $market  --topk 5 --ndrop 2 --label $label --ht $ht --feat $feat --retrain --savecsv > ./log/$dt.log
dt=$(date +"%Y%m%d_%H%M%S")
python ./src/bench.py --expname $expname_$dt --benchdataset $benchdataset --market $market  --topk 10 --ndrop 1 --label $label --ht $ht --feat $feat --retrain --savecsv > ./log/$dt.log
dt=$(date +"%Y%m%d_%H%M%S")
python ./src/bench.py --expname $expname_$dt --benchdataset $benchdataset --market $market  --topk 10 --ndrop 2 --label $label --ht $ht --feat $feat --retrain --savecsv > ./log/$dt.log
dt=$(date +"%Y%m%d_%H%M%S")
python ./src/bench.py --expname $expname_$dt --benchdataset $benchdataset --market $market  --topk 20 --ndrop 1 --label $label --ht $ht --feat $feat --retrain --savecsv > ./log/$dt.log
dt=$(date +"%Y%m%d_%H%M%S")
python ./src/bench.py --expname $expname_$dt --benchdataset $benchdataset --market $market  --topk 20 --ndrop 2 --label $label --ht $ht --feat $feat --retrain --savecsv > ./log/$dt.log
dt=$(date +"%Y%m%d_%H%M%S")
python ./src/bench.py --expname $expname_$dt --benchdataset $benchdataset --market $market  --topk 20 --ndrop 5 --label $label --ht $ht --feat $feat --retrain --savecsv > ./log/$dt.log
dt=$(date +"%Y%m%d_%H%M%S")
python ./src/bench.py --expname $expname_$dt --benchdataset $benchdataset --market $market  --topk 50 --ndrop 1 --label $label --ht $ht --feat $feat --retrain --savecsv > ./log/$dt.log
dt=$(date +"%Y%m%d_%H%M%S")
python ./src/bench.py --expname $expname_$dt --benchdataset $benchdataset --market $market  --topk 50 --ndrop 2 --label $label --ht $ht --feat $feat --retrain --savecsv > ./log/$dt.log
dt=$(date +"%Y%m%d_%H%M%S")
python ./src/bench.py --expname $expname_$dt --benchdataset $benchdataset --market $market  --topk 50 --ndrop 5 --label $label --ht $ht --feat $feat --retrain --savecsv > ./log/$dt.log
done
done
done
done
done