# Databricks notebook source
import csv
import pandas as pd

names_csv = pd.read_csv("names.csv")

names_dictionary = {}

for i, row in names_csv.iterrows():
    nicknames = [name for name in row[1:] if isinstance(name, str)]
    names_dictionary[row[0]] = nicknames

print(names_dictionary)