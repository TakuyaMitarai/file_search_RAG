import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb
from langchain.document_loaders import PDFLoader, TextLoader, PythonLoader, CppLoader


# モデルの読み込みと量子化
def load_model():
    model_name = "umiyuki/Umievo-itr012-Gleipnir-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb.QuantizationConfig(bits=4)
    )
    return model, tokenizer


# ディレクトリ内のファイルを再帰的にリストアップ
def list_files(directory):
    file_types = (".pdf", ".py", ".cpp", ".txt")
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(file_types):
                files.append(os.path.join(root, filename))
    return files


# ファイルを読み込んで内容を取得
def read_file(file_path):
    extension = os.path.splitext(file_path)[1]
    if extension == ".pdf":
        loader = PDFLoader(file_path)
    elif extension == ".py":
        loader = PythonLoader(file_path)
    elif extension == ".cpp":
        loader = CppLoader(file_path)
    else:
        loader = TextLoader(file_path)

    document = loader.load()
    return document.text


# LLMに要約を生成させる
def summarize(text, model, tokenizer):
    prompt = f"次の文章やコードを要約してください．以下内容\n{text}"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=512)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# 結果をファイルに書き込む
def write_summary(file_summaries, output_file):
    with open(output_file, "w") as f:
        for summary, file_path in file_summaries:
            f.write(f"{summary}\nファイルのパスは {file_path} です\n\n")


# メイン処理
def main(directory):
    model, tokenizer = load_model()
    files = list_files(directory)
    file_summaries = []

    for file_path in files:
        content = read_file(file_path)
        summary = summarize(content, model, tokenizer)
        file_summaries.append((summary, file_path))

    output_file = os.path.join(directory, "directory_summary.txt")
    write_summary(file_summaries, output_file)


if __name__ == "__main__":
    target_directory = "/path/to/your/directory"  # 対象のディレクトリを指定
    main(target_directory)
