TASK=$1
OUTPUT_PATH=$2
RETRIEVER_PATH=$3
python rag_testing_${TASK}.py --retriever_trained --model_modified --output_path $OUTPUT_PATH --trained_retriever_path $RETRIEVER_PATH
