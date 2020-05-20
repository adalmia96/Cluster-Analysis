
for em in word2vec fasttext; do

  python3 code/score.py --entities $em --clustering_algo VMFM --doc_info WGT --rerank freq > results/$em-VMFM-weighted-rr.txt

  python3 code/score.py --entities $em --clustering_algo VMFM --doc_info WGT > results/$em-VMFM-weighted.txt

  python3 code/score.py --entities $em --clustering_algo VMFM --rerank freq > results/$em-VMFM-rr.txt

  python3 code/score.py --entities $em --clustering_algo VMFM > results/$em-VMFM.txt

done
