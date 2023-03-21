import numpy as np
import streamlit as st
import tensorflow as tf
import csv
from AirWriting import predictUsingModel

# predictUsingModel()




def webpage():
    top_classes,top_probs= predictUsingModel()

    for i in range(3):
        st.title(f"{tamilCharacterCode[top_classes[i]]}: {top_probs[i]}")
        st.title(f"{tamilCharacterCode[top_classes[i]]}")
    
    


# runs only if we run from a command prompt
if __name__=='__main__':


    tamilCharacterCode = []

    w,h=128,128
    with open('unicodeTamil.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        for i in data:
            go = i[1].split(' ')
            charL = ""
            for gg in go:
                charL = charL + "\\u"+str(gg)
            tamilCharacterCode.append(charL.encode('utf-8').decode('unicode-escape'))

    webpage()
