from bs4 import BeautifulSoup
from llmbot import BotActor

book = "/home/i/Downloads/FB2/СНЕГОВ – Люди как боги (1992).fb2"

with open(book, "rb") as file:
    soup = BeautifulSoup(file, "xml")

print(soup.original_encoding)

# Get Title (using .find to avoid errors if missing)
title_tag = soup.find("book-title")
if title_tag:
    print(f"Title: {title_tag.get_text()}\n")


# Get Body Text (paragraphs)
paragraphs = [p.get_text() for p in soup.find_all("p")]
print("\n".join(paragraphs))


bot = BotActor()

for text in paragraphs:
    bot.speak(text)
