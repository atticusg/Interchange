DATE=$1
TIME=$(date +"%m%d%H%M%S")
rm experiment_data/bert/bert-$DATE-*.db
scp "hansonlu@sc.stanford.edu:Interchange/experiment_data/bert/bert-$DATE.db" "experiment_data/bert/bert-$DATE-$TIME.db"
echo "fetched latest database bert-$DATE.db to:"
echo "experiment_data/bert/bert-$DATE-$TIME.db"
