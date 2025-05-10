from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import render_template

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("prompt", "")
    system_instruction = "شما یک دستیار فارسی هستید که به سؤالات با لحن دوستانه پاسخ می‌دهید."
    full_prompt = system_instruction + "\n\n" + user_input

    inputs = tokenizer(full_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded[len(full_prompt):].strip()

    return jsonify({"response": response})

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-fa")
    model = AutoModelForCausalLM.from_pretrained("HooshvareLab/gpt2-fa")
    app.run(debug=True)
