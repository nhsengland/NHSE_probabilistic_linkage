# Databricks notebook source
def dict_from_class(cls):
  return dict(
         (key, value)
         for (key, value) in cls.__dict__.items()
         )