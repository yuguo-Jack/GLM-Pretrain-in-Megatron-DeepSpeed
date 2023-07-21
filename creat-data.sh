python3 tools/preprocess_data.py \
       --input /work/home/yuguo960516yuguo/llm/data/oscar-1GB.jsonl \
       --output-prefix /work/home/yuguo960516yuguo/llm/data_output/my_glm \
       --dataset-impl mmap \
       --tokenizer-type IceTokenizer \
       --append-eod \
       --workers 8
