import os
import logging 
import json


chemin_courant = os.path.dirname(__file__)
chemin_ = os.path.join(chemin_courant, "data", "articles.json")
print(chemin_)
class ListeCourse:

    def __init__(self, title) -> None:
        self.title = title.title()

    def __str__(self) -> str:
        return self.title
    
    def _get_article(self):
        with open(chemin_, "r") as f:
            return json.load(f)


    def _write_article(self, article):
        with open(chemin_, "w") as f:
            json.dump(article, f , indent=4)

    def add_to_articles(self):
        liste_article = self._get_article()

        if self.title not in liste_article:
            liste_article.append(self.title)
            self._write_article(liste_article)
            return True
        else: 
            logging.warning(f"l'article {self.title} est déjà enregistré")
            return False

    def remove_from_articles(self):
        liste_articles = self._get_article()

        if self.title in liste_articles:
            liste_articles.remove(self.title)
            self._write_article(liste_articles)

def get_articles():
    with open(chemin_, "r") as f:
        noms_articles = json.load(f)

    articles = [ListeCourse(article) for article in noms_articles]
    return articles

if __name__ == "__main__":
    #m = ListeCourse("mangue")
    m = ListeCourse("Banane")
    print(m.add_to_articles())
