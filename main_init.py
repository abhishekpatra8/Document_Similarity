from flask import Flask, request, render_template, Response
import document_check as dc, json

app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/', methods=['GET', 'POST'])
def index():
    # return render_template('index.html', display=0)


# @app.route('/check', methods=['POST'])
# def result():
    val = []
    if request.method == 'POST':
        data = request.get_json()
        # print(data['doc1'], data['doc2'])
        # return "1"
        doc, output = dc.text_processing(data['doc1'].lower(), data['doc2'].lower())
        v1, v2, v3 = dc.string_comparater(doc)
        val.append(output[0])
        val.append(output[1])
        val.append(v1)
        val.append(v2)
        val.append(v3)
        acc_val = dc.check_accuracy(val)
        if acc_val == 0:
            final_s = dc.check_similarity(data['doc1'], data['doc2'])
            val.append(final_s[0])
            val.append(final_s[1])
        else:
            val.append('-')
            val.append('-')
        print(len(output), output, acc_val)
        result = json.dumps({'success': 1, 'result1': data['doc1'], 'result2': data['doc2'], 'output1': str(val[0]), 'output2': str(val[1]), 'accuracy': str(acc_val)})
        return Response(result, status=200, mimetype="application/json")
        # return render_template("index.html", display=1, len=len(output), result=data, output=val, accuracy=acc_val)
    else:
        return render_template('index.html', display=0)


if __name__ == "__main__":
    # app.run("0.0.0.0", 81)
    app.run("localhost", 81)
