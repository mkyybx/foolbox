#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as ms  # import cross_val_scores
import csv
import os
import pickle


events = ["NetworkLinkRequest", "NetworkScriptRequest", "NetworkIframeRequest",
          "NetworkImageRequest", "NetworkXMLHTTPRequest", "NetworkVideoRequest"]
# tags = ['UNKNOWN','UNCOMMON','HEAD','BODY','DIV','VIDEO','FORM','HTML','A','SPAN','SECTION','LI','PICTURE','UL','MAIN','HEADER','TD','FOOTER','P','FIGURE','NAV','CENTER','B','ASIDE','IFRAME','DT','INS','MAINCONTENT','H4','SCRIPT','LEFT','STRONG','ARTICLE','URSCRUB','FONT','I','H1','LABEL','H2','PRE','BUTTON','head','body','NOINDEX','TR','A-IMG','TABLE','NOSCRIPT','SMALL','ISGOOGLEREMARKETING','BODYCLASS=HOME','TH','TERMS-BUTTON','W7-COMPONENT','NOLAYER','MYBASE','DL','HOME-PAGE','TEXT','FRAGMENT','CUSTOMHTML','RA','TBODY','LINK','BASE','META','STYLE','IMG','TITLE','SOURCE','INPUT','COOKIEJAR','BR','CODE','FB:LIKE','CLOUDFLARE-APP','svg','H3','CANVAS','AUDIO','H5','TWC-NON-EXPANDABLE-WIDGET','APP-ROOT','VT-APP','ENVIRONMENT-OVERLAY','APP','GTM','ONL-ROOT','COOKIE-CONSENT','PHOENIX-SCRIPT','PHOENIX-PAGE','OBJECT','IAS-AD','FBTRASH','BLOCKQUOTE','YMAPS','APM_DO_NOT_TOUCH','WB:FOLLOW-BUTTON','script','UI-VIEW','ICON','CUSTOM-STYLE','DASHI-SERVICE-WORKER','DASHI-ANALYTICS-TAG','ABBR','AMP-GEO','ALPS-ACCOUNT-PANEL','IMIMXXXYYY','AMP-CONSENT','W-DIV','CFC-APP-ROOT','DASHI-LINK-SCANNER']

tag_1 = ["HEAD", "?", "BODY", "DIV", "VIDEO", "FORM", "ASIDE", "TD", "SPAN", "A", "TR", "SECTION", "DT", "EU8H8PT9X2", "TELPCG9PCDGTVD", "SHFMKILCCDM7XX", "LI", "P", "STRONG", "HEADER", "HTML", "INS", "ITEM", "H1", "MAIN", "CENTER", "PICTURE", "UL", "FOOTER", "BUTTON", "NOLAYER", "YATAG", "FIGURE", "FONT", "NOINDEX", "B", "ARTICLE", "IE:HOMEPAGE", "BLOCKQUOTE", "NAV", "H2", "H3", "SYSTEM-REGION", "B-AVATAR", "NOBR", "TABLE", "h:head", "div", "DATA", "WRAPPER", "SMALL", 'STYLE=".TS_1.OB-VERTICAL-STRIP-LAYOUT', "SERVERS", "IMG", "EM",
         "NG-INCLUDE", "FIELDSET", "DD", 'SRC="IMAGES', "HOME-PAGE", "OBJECT", "MATA", "PHOENIX-PAGE", "head", "body", "OC-COMPONENT", "LABEL", "FRAGMENT", "NTV-DIV", "EA-SECTION", "EA-HERO", "W7-COMPONENT", "APM_DO_NOT_TOUCH", "HINETWORK", "APPINFO", "H4", "DL", "C:IF", "CUSTOM", "HERE", "I", "CODE", "RIGHT_BANNER", "LEFT_BANNER", "NRK-BOTTOM-MENU", "RA", "MKT-HERO-MARQUEE", "KANHANBYPASS", "TEXT", "SCRIPT", "WB:FOLLOW-BUTTON", "LEFT", "TMO-DIGITAL-HEADER", "TMO-DIGITAL-FOOTER", "UP-TRACK", "WIX-IMAGE", "YT-IMG-SHADOW", "FJTIGNOREURL"]
tag_2 = ["?", "UNKNOWN", "SCRIPT", "IFRAME", "STYLE", "META", "DIV", "LINK", "FORM", "A", "IMG", "TITLE", "INS", "FOOTER", "TABLE", "SPAN", "NOSCRIPT", "SECTION", "BODY", "YM-MEASURE", "HEADER", "H1", "ARTICLE", "UL", "BASE", "NAV", "SOURCE", "svg", "STRONG", "MAIN", "BR", "HR", "INPUT", "CENTER",
         "ION-APP", "SMALL", "HEAD", "H5", "H4", "VIDEO", "BUTTON", "OBJECT", "H2", "BLOCKQUOTE", "PHOENIX-SCRIPT", "ASIDE", "P", "TEMPLATE", "MODAL-MANAGER", "script", "iframe", "APP-ROOT", "NOINDEX", "ALPS-ACCOUNT-PANEL", "AUDIO", "FEEFOWIDGET-CONTAINER-FLOATING-SERVICE", "EMBED", "DAC-IVT-OGV", "H3"]
tag_3 = ["UNKNOWN", "HEAD", "HTML", "BODY", "DIV", "SPAN", "CENTER", "TD", "NOINDEX", "?", "UL", "LI", "FORM", "ASIDE", "FOOTER", "A", "WB:SHARE-BUTTON", "STRONG", "SECTION", "ARTICLE", "HEADER", "VIDEO-JS", "P", "SYSTEM-REGION", "INS", "SCRIPT", "WB:FOLLOW-BUTTON",
         "FIGURE", "PHOENIX-SCRIPT", "head", "body", "FRAGMENT", "MAIN", "APM_DO_NOT_TOUCH", "YATAG", "APPINFO", "CUSTOM", "I", "WEB-REQUEST-TRACKER", "CONTENT", "APESTER-LAYER", "CLOUDFLARE-APP", "ESI:INCLUDE", "NTV-DIV", "TMO-DIGITAL-FOOTER", "HINETWORK"]
tag_4 = ["UNKNOWN", "SCRIPT", "BODY", "?", "LINK", "IFRAME", "DIV", "STYLE", "META", "IMG", "BASE", "SPAN", "CENTER", "TITLE", "NOSCRIPT", "VIDEO", "H1", "A", "UL", "HEADER", "SECTION", "CLOUDFLARE-APP", "INS", "FOOTER", "BR", "BUTTON", "EM", "MAIN", "FORM", "INPUT", "TABLE", "VIDEO-JS", "P", "HEAD", "BLOCKQUOTE",
         "I", "PICTURE", "WEBENGAGEDATA", "NAV", "MATA", "LI", "H5", "PSANODE", "script", "svg", "AMP-CONSENT", "APP-ROOT", "TEMPLATE", "OBJECT", "FDJ-AUTHENTICATION", "AUDIO", "YATAG", "COOKIEJAR", "CUSTOM", "H4", "NRK-BOTTOM-MENU", "URSCRUB", "ESI:INCLUDE", "ASIDE", "TMO-DIGITAL-HEADER", "DOM-MODULE", "FONT", "HINETWORK"]


def setup_clf(pickle_path):
    with open(pickle_path, 'rb') as fin:
        clf = pickle.load(fin)
    return clf


def transform_row(row):
    global events, tag_1, tag_2, tag_3, tag_4

    row[13] = events.index(row[13])

    if row[23] in tag_1:
        row[23] = tag_1.index(row[23])
    elif row[23].strip() == '':
        row[23] = 0
    else:
        row[23] = 1

    if row[26] in tag_2:
        row[26] = tag_2.index(row[26])
    elif row[26].strip() == '':
        row[26] = 0
    else:
        row[26] = 1

    if row[42] in tag_3:
        row[42] = tag_3.index(row[42])
    elif row[42].strip() == '':
        row[42] = 0
    else:
        row[42] = 1

    if row[45] in tag_4:
        row[45] = tag_4.index(row[45])
    elif row[45].strip() == '':
        row[45] = 0
    else:
        row[45] = 1

    row[4] = round(float(row[4]), 3)
    row[5] = round(float(row[5]), 3)
    row[10] = round(float(row[10]), 3)
    row[32] = round(float(row[32]), 3)
    row[51] = round(float(row[51]), 3)

    return row


def predict(x, clf):
    TO_EXCLUDE = {0, 1, 9, 31, 50}
    if len(x) == 65:
        x_new = []
        cnt = 0
        for i in range(70):
            if i in TO_EXCLUDE:
                x_new.append("dummy")
            else:
                x_new.append(x[cnt])
                cnt += 1
        x = x_new
    transformed_x = transform_row(x)
    # [:-1] to remove class
    # print(transformed_x)
    trimmed_x = [element for i, element in enumerate(transformed_x) if i not in TO_EXCLUDE]
    trimmed_x = np.array([trimmed_x])
    res = clf.predict(trimmed_x)
    return res


# For testing
if __name__ == "__main__":
    clf = setup_clf("/home/shitong/Desktop/AdGraphAPI/scripts/model/rf.pkl")
    x = "https://www.cnn.com/,URL_1,13,12,1.083333,0.923077,1,0,1,248.922208,0.285714,0,0,NetworkLinkRequest,0,0,0,0,0,0,0,0,0,HEAD,1,6,META,0,1,1,2,273.814429,0.571429,0,0,0,0,0,0,0,0,0,UNKNOWN,0,0,UNKNOWN,0,0,0,0,0.000000,0.000000,0,0,0,0,0,0,0,0,0,0,1,0,0,0,45,0,1,1,AD".split(
        ',')
    print(predict(x, clf))
# clf = RandomForestClassifier(n_estimators = 100, max_depth = None, random_state = 1, criterion = "entropy") # n_estimators is numTree. max_features is numFeatures
# clf.fit(np.asarray(testdata),np.asarray(labels))
