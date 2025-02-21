#!/bin/bash

source ~/.bashrc 
conda activate qlib39
cd /home/linq/finance/qniverse

ST='2024-10-15'

echo '=================START======================'
echo $(date)

ED=$(python ./reporter/get_date.py)
# ED='2025-01-06'

if [ $ED == "0" ]; then
# if [ '1' == "0" ]; then
    echo "No new data. Have a nice day!"
else
    echo "New data detected: $ED"
    echo 'Update QLIB Database'
    python ./reporter/data_update.py
    python ~/finance/LiLab/lilab/qlib/scripts/dump_bin.py dump_all --csv_path /data/linq/.qlib/qlib_data/cn_data/cn_1d_norm --qlib_dir /data/linq/.qlib/qlib_data/cn_data --freq day  --exclude_fields date,symbol
    echo 'QLIB Database updated'

    # python -m lilab.qlib.private.bensemble --btstart $ST
    python -m lilab.qlib.private.mensemble --btstart $ST
    python -m lilab.qlib.private.mensemble0 --btstart $ST
    # python -m lilab.qlib.private.mensemblel0 --btstart $ST
    # python -m lilab.qlib.private.mensembleh0 --btstart $ST
    python -m lilab.qlib.private.nensemble --btstart $ST
    python -m lilab.qlib.private.densemble --btstart $ST
    # python reporter/model/gdbt_skip_pred.py
    # python reporter/model/gdbt_skip_fig.py --btstart $ST

    echo 'push to github...'
    git add .
    git commit -m $ED
    git push
    
    cd /home/linq/finance/private_trade
    jupyter nbconvert --to notebook --execute --inplace ensemble_private.ipynb
    # git add .
    # git commit -m $ED
    # git push

    # cd /home/linq/finance/qniverse
    # echo 'Running baselines...'
    # python ./TSLib/train_today.py --config_file ./configs/config_timesnet.yaml  --btstart $ST
    # python ./TSLib/train_today.py --config_file ./configs/config_patchtst.yaml --btstart $ST
    # python ./TSLib/train_today.py --config_file ./configs/config_pdf.yaml --btstart $ST
    # python ./TSLib/train_today.py --config_file ./configs/config_segrnn.yaml  --btstart $ST
    # python ./TSLib/train_today.py --config_file ./configs/config_timebridge.yaml  --btstart $ST
    # python ./TSLib/train_today.py --config_file ./configs/config_timemixer.yaml  --btstart $ST
    # python ./TSLib/train_today.py --config_file ./configs/config_wftnet.yaml  --btstart $ST
    # python reporter/model/gdbt_pred.py
    # python reporter/model/gdbt_fig.py --btstart $ST
    # python reporter/model/xgb.py --btstart $ST
    # python reporter/model/olmar.py --btstart $ST --btend $ED
    # python reporter/model/kelly.py --btstart $ST --btend $ED
    # python reporter/model/up.py --btstart $ST --btend $ED
    # python reporter/model/ons.py --btstart $ST --btend $ED
    # python reporter/model/anticor.py --btstart $ST --btend $ED
    # python reporter/model/corn.py --btstart $ST --btend $ED
    # python reporter/model/bnn.py --btstart $ST --btend $ED
    # python reporter/model/pamr.py --btstart $ST --btend $ED
    # python reporter/model/wmamr.py --btstart $ST --btend $ED
    # python reporter/model/mpt.py --btstart $ST --btend $ED
    # python reporter/model/eg.py --btstart $ST --btend $ED
    # python reporter/model/tco.py --btstart $ST --btend $ED

    # echo 'push to github...'
    # git add .
    # git commit -m $ED
    # git push

    
    # cd /home/linq/finance/private_trade
    jupyter nbconvert --to notebook --execute --inplace ensemble_main.ipynb
    git add .
    git commit -m $ED
    git push

fi