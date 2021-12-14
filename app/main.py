
from flask import render_template, request, flash, redirect, url_for
from app import app
from werkzeug.utils import secure_filename
import os


@app.route('https://convertimagetotable.herokuapp.com/')
def home():
    return render_template('index.html')

@app.route('https://convertimagetotable.herokuapp.com/con')
def con():
    return render_template('con.html')

ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('https://convertimagetotable.herokuapp.com', methods=['POST'])
def uploadImg():
    if 'file' not in request.files:
        flash('No File Attached !')
    file = request.files['file']
    file.filename = "image.png"


    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        conversion()

        return render_template('index.html', filename=filename)

    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('https://convertimagetotable.herokuapp.com/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

def conversion():
    import cv2
    import pytesseract as tess
    from pytesseract import Output
    import pandas as pd
    tess.pytesseract.tesseract_cmd = r'C:\Users\bishn\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

    img = cv2.imread('static/uploads/image.png')
    img = cv2.resize(img, (int(img.shape[1] + (img.shape[1] * .1)),
                           int(img.shape[0] + (img.shape[0] * .25))),
                     interpolation=cv2.INTER_AREA)

    #imgColor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    custom_config = r'-l eng --oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:.$%./@& *"'
    d = tess.image_to_data(img, config=custom_config, output_type=Output.DICT)
    df = pd.DataFrame(d)

    # clean up blanks
    df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # sort blocks vertically
    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
    for block in sorted_blocks:
        curr = df1[df1['block_num'] == block]
        sel = curr[curr.text.str.len() > 3]
        # sel = curr
        char_w = (sel.width / sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        text = ''
        for ix, ln in curr.iterrows():
            # add new line when necessary
            if prev_par != ln['par_num']:
                text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                text += '\n'
                prev_line = ln['line_num']
                prev_left = 0

            added = 0  # num of spaces that should be added
            if ln['left'] / char_w > prev_left + 1:
                added = int((ln['left']) / char_w) - prev_left
                text += ',' + '' * added
            text += ln['text'] + ' '
            prev_left += len(ln['text']) + added + 1

        text=text.replace(',',', ')
        print(text)


        with open('test.csv', 'w') as f:
            f.write(text)
        df = pd.read_csv('test.csv', error_bad_lines=False)
        df = df.fillna('')

        a = df.columns.values.tolist()
        print(a)
        if (a[0] == 'Unnamed: 0'):
            del df['Unnamed: 0']

        print(df)
        html_table = df.to_html()

        f = open('templates/con.html', 'w')
        f.write(html_table)
        f.close()



