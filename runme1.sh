nohup python3 -u -m nel.main --mode prerank  --multi_instance --n_negs 5 --preranked_data /home/wenh/data/generated/test_train_data/preranked_all_datasets_10kRCV1_large --n_not_inc 5 --n_docs 500000 > /home/wenh/deep-wnel/run.log 2>&1 &

