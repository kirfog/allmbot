#!/usr/bin/env python3
import argparse

from bs4 import BeautifulSoup

from llmbot import BotActor

parser = argparse.ArgumentParser(description="fb2 reader")

parser.add_argument(
    "book",
    type=str,
    help="thebook.fb2",
)

args = parser.parse_args()

with open(args.book, "rb") as file:
    soup = BeautifulSoup(file, "xml")

print(soup.original_encoding)

title_tag = soup.find("book-title")
if title_tag:
    print(f"Title: {title_tag.get_text()}\n")

paragraphs = [p.get_text() for p in soup.find_all("p")]

bot = BotActor()

for text in paragraphs:
    print(text)
    bot.speak(text)
