#!/usr/bin/env python
# coding=utf-8
#
# Basic Python script to OCR Hoogvliet nutrient images.
#
# Requires tesseract 3.04 or higher.
#
import re
import sys
import urllib2
import cStringIO
from PIL import Image
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile

# scale up image before processing
scale = 3
# image thresholding value
threshold = 0.65

# tesseract language
lang = 'nld'

# help tesseract to detect the right words
user_words = '''
    Kilojoule Kilocalorie Gram Milligram Microgram Milliliter
    alcohol Biotine Eiwitten Energie Foliumzuur Fosfor IJzer Kalium Koolhydraten
    Natrium Niacine Magnesium Panthotheenzuur Polyolen Riboflavine Suikers Thiamine
    Vetten Voedingsvezels Vetzuren Vitamine Zout acetate Cinnamyl enkelvoudig
    meervoudig verzadigd onverzadigd totaal internationale tabel GDSN
'''.strip().split()

# fix some common misdetections (order is important here)
replacements = {
    "calon'e":                     'calorie',
    r'[|Il][Gﬁ][|Il]o':            'Kilo',
    u'ﬂ':                          'fl',
    r'V[rn]tamine':                'Vitamine',
    u'Vúamine':                    'Vitamine',
    r'N[ir]tamine':                '/Vitamine',
    'Mtamine':                     '/Vitamine',
    'Panthotheenzuu(an)?th?amine': 'Pantotheenzuur/Vitamine',
    u'Vitamine Bô':                'Vitamine B6',
    r'Vitamine [38]([0-9])':       r'Vitamine B\1',
    r'Vitamine [38B]S':            'Vitamine B5',
    "Natn'um":                     'Natrium',
    'intemationale':               'internationale'
}

def check():
    '''Return whether the dependencies are available'''
    try:
        proc = Popen(['tesseract', '-v'], stderr=PIPE)
        output = proc.stderr.read().decode('utf-8')
        proc.wait()
        return "tesseract" in output
    except OSError:
        return False

def opener(filename_or_url):
    if re.match(r'^(http|https|ftp):', filename_or_url, flags=re.IGNORECASE):
        return urllib2.urlopen(sys.argv[1])
    else:
        return open(filename_or_url, 'rb')

def tesseract(img):
    '''Return OCR-ed text from image by running tesseract'''
    global lang
    global user_words
    with NamedTemporaryFile(prefix='tess_', suffix='.bmp') as imgfile:
        with NamedTemporaryFile(prefix='tess_', suffix='.words') as wordsfile:
            img.save(imgfile)
            wordsfile.write('\n'.join(user_words))
            wordsfile.flush()
            command = ['tesseract', '-psm', '4', '-l', lang, '--user-words', wordsfile.name, imgfile.name, 'stdout']
            proc = Popen(command, stdout=PIPE)
            output = proc.stdout.read().decode('utf-8')
            proc.wait()
            return output

def preprocess(img):
    '''Return pre-processed image'''
    global scale
    global threshold
    img = img.convert('L') # grayscale
    img = img.resize((img.size[0] * scale, img.size[1] * scale), Image.ANTIALIAS)
    img = img.point(lambda i: i > threshold * 255 and 255)
    return img

def postprocess(txt):
    '''Post-process text'''
    global replacements
    for frm, to in replacements.iteritems():
        txt = re.sub(frm, to, txt, flags=re.IGNORECASE)
    return txt

if __name__ == '__main__':
    if sys.argv[1] == 'check':
        sys.exit(0 if check() else 1)
    else:
        img = Image.open(opener(sys.argv[1]))
        txt = tesseract(preprocess(img))
        print postprocess(txt)

