#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as ms #import cross_val_scores
import csv
import os
import pickle


# # data_file = '/mnt/drive/work/sample_rows.csv'
# home_dir = os.getenv("HOME")
# # home_dir = "/run/media/shitong/4B06DD0D70B46233"
# base_directory = home_dir + '/Desktop/'
# data_file = base_directory + 'unique_features_data.csv'
events = ['NetworkLinkRequest', 'NetworkScriptRequest', 'NetworkImageRequest', 'NetworkIframeRequest', 'NetworkXMLHTTPRequest', 'NetworkVideoRequest']
# tags = ['UNKNOWN','UNCOMMON','HEAD','BODY','DIV','VIDEO','FORM','HTML','A','SPAN','SECTION','LI','PICTURE','UL','MAIN','HEADER','TD','FOOTER','P','FIGURE','NAV','CENTER','B','ASIDE','IFRAME','DT','INS','MAINCONTENT','H4','SCRIPT','LEFT','STRONG','ARTICLE','URSCRUB','FONT','I','H1','LABEL','H2','PRE','BUTTON','head','body','NOINDEX','TR','A-IMG','TABLE','NOSCRIPT','SMALL','ISGOOGLEREMARKETING','BODYCLASS=HOME','TH','TERMS-BUTTON','W7-COMPONENT','NOLAYER','MYBASE','DL','HOME-PAGE','TEXT','FRAGMENT','CUSTOMHTML','RA','TBODY','LINK','BASE','META','STYLE','IMG','TITLE','SOURCE','INPUT','COOKIEJAR','BR','CODE','FB:LIKE','CLOUDFLARE-APP','svg','H3','CANVAS','AUDIO','H5','TWC-NON-EXPANDABLE-WIDGET','APP-ROOT','VT-APP','ENVIRONMENT-OVERLAY','APP','GTM','ONL-ROOT','COOKIE-CONSENT','PHOENIX-SCRIPT','PHOENIX-PAGE','OBJECT','IAS-AD','FBTRASH','BLOCKQUOTE','YMAPS','APM_DO_NOT_TOUCH','WB:FOLLOW-BUTTON','script','UI-VIEW','ICON','CUSTOM-STYLE','DASHI-SERVICE-WORKER','DASHI-ANALYTICS-TAG','ABBR','AMP-GEO','ALPS-ACCOUNT-PANEL','IMIMXXXYYY','AMP-CONSENT','W-DIV','CFC-APP-ROOT','DASHI-LINK-SCANNER']

tag_1 = ['HEAD','BODY','DIV','VIDEO','FORM','HTML','IE:HOMEPAGE','A','SPAN','SECTION','LI','PICTURE','UL','MAIN','HEADER','TD','FOOTER','P','FIGURE','NAV','CENTER','B','ASIDE','IFRAME','DT','INS','MAINCONTENT','H4','SCRIPT','LEFT','STRONG','ARTICLE','URSCRUB','FONT','I','ESI:TEXT','H1','YT-IMG-SHADOW','FJTIGNOREURL','FIELDSET','CNX','DD','LABEL','H2','PRE','BUTTON','head','body','NOINDEX','TR','A-IMG','EM','OC-COMPONENT','TMPL_IF','TABLE','NOSCRIPT','YATAG','HINETWORK','LINKÂ HREF=','METAÃ','SMALL','ISGOOGLEREMARKETING','BODYCLASS=HOME','TH','OLANG','C:IF','Pæ¥æ¬éç¨®ä¸ç­çé­çæ°ææè¿å çºèç¨±æ¥æ¬25å¹´ä¾çæå¼·é¢±é¢¨ãçå­ãä¸é¸è®ä¸å°å¬å¸ç¼åºï¼å¸°å®å½ä»¤ï¼','TERMS-BUTTON','W7-COMPONENT','METAÂ NAME=CXENSEPARSE:URLÂ CONTENT=HTTP:','MKT-HERO-MARQUEE','NOLAYER','METAâ','MYBASE','DL','HOME-PAGE','CFC-IMAGE','TEXT','WAINCLUDE','METAÂ PROPERTY=FB:PAGES','FRAGMENT','GWC:HIT','CUSTOMHTML','ESI:INCLUDE','RA','TBODY','HTMLÂ XMLNS=HTTPS:','VNN']
tag_2 = ['UNKNOWN','UNCOMMON','SCRIPT','LINK','IFRAME','BASE','META','DIV','STYLE','BODY','IMG','NOSCRIPT','A','HEADER','VIDEO','TITLE','RMFNLJMLWDDURTPYLAMWH','FOOTER','SOURCE','MAIN','INPUT','SPAN','SECTION','COOKIEJAR','BR','URSCRUB','FORM','P','UL','CODE','FB:LIKE','NAV','LI','CLOUDFLARE-APP','svg','H2','H3','CANVAS','AUDIO','H5','TABLE','TWC-NON-EXPANDABLE-WIDGET','H4','CENTER','INS','APP-ROOT','HEAD','VT-APP','ENVIRONMENT-OVERLAY','I','APP','GTM','ONL-ROOT','MYBASE','COOKIE-CONSENT','PHOENIX-SCRIPT','PHOENIX-PAGE','OBJECT','IAS-AD','BUTTON','EM','FBTRASH','BLOCKQUOTE']
tag_3 = ['UNKNOWN','HEAD','DIV','BODY','SPAN','UL','FORM','LI','ASIDE','P','SECTION','HTML','TBODY','A','ESI:TEXT','H1','FJTIGNOREURL','CLOUDFLARE-APP','CNX','B','FOOTER','BUTTON','INS','CENTER','YATAG','SCRIPT','NOINDEX','TD','HEADER','head','body','HINETWORK','MAIN','ARTICLE','YMAPS','IFRAME','NOSCRIPT','FONT','FIELDSET','H3','FIGURE','ESI:INCLUDE','Pæ¥æ¬éç¨®ä¸ç­çé­çæ°ææè¿å çºèç¨±æ¥æ¬25å¹´ä¾çæå¼·é¢±é¢¨ãçå­ãä¸é¸è®ä¸å°å¬å¸ç¼åºï¼å¸°å®å½ä»¤ï¼','DT','MYBASE','PHOENIX-SCRIPT','APM_DO_NOT_TOUCH','SMALL','FRAGMENT','WB:FOLLOW-BUTTON','TH','NAV','HTMLÂ XMLNS=HTTPS:']
tag_4 = ['LI','DL','P','ASIDE','INS','HEAD','script','UI-VIEW','ICON','CUSTOM-STYLE','BASE','svg','I','NOINDEX','DASHI-SERVICE-WORKER','DASHI-ANALYTICS-TAG','ABBR','CANVAS','AMP-GEO','H2','ESI:INCLUDE','ALPS-ACCOUNT-PANEL','IMIMXXXYYY','NAV','PHOENIX-SCRIPT','AMP-CONSENT','W-DIV','CFC-APP-ROOT','URSCRUB','CENTER','H3','DASHI-LINK-SCANNER','H4']


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
    transformed_x = transform_row(x)
    # [:-1] to remove class
    trimmed_x = [element for i, element in enumerate(transformed_x[:-1]) if i not in TO_EXCLUDE]
    trimmed_x = np.array([trimmed_x])
    res = clf.predict(trimmed_x)
    return res

# For testing
if __name__ == "__main__":
    clf = setup_clf("/home/shitong/Desktop/AdGraphAPI/scripts/model/rf.pkl")
    x = "https://www.cnn.com/,URL_1,13,12,1.083333,0.923077,1,0,1,248.922208,0.285714,0,0,NetworkLinkRequest,0,0,0,0,0,0,0,0,0,HEAD,1,6,META,0,1,1,2,273.814429,0.571429,0,0,0,0,0,0,0,0,0,UNKNOWN,0,0,UNKNOWN,0,0,0,0,0.000000,0.000000,0,0,0,0,0,0,0,0,0,0,1,0,0,0,45,0,1,1,AD".split(',')
    print(predict(x, clf))
# clf = RandomForestClassifier(n_estimators = 100, max_depth = None, random_state = 1, criterion = "entropy") # n_estimators is numTree. max_features is numFeatures
# clf.fit(np.asarray(testdata),np.asarray(labels))
