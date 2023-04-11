import argparse
import calendar
import csv
import pathlib
from pathlib import Path
import re
from datetime import date, timedelta
from time import sleep

from bs4 import BeautifulSoup
from numpy.random import uniform
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By


def get_next_url(soup):
    for review in soup.find_all("a", {"class": "ui_button"}):
        # print(review)
        if review.text == "Next":
            try:
                return "https://www.tripadvisor.com" + review.get("href")
            except:
                break

def get_reviewer_name_date(soup):
    date_pub = []
    commenters = []
    
    for review in soup.find_all("div", {"class": "cRVSd"}):
        if bool(re.search("wrote a review", str(review.text))):
            name, date = review.text.split(" wrote a review ")
                        
            if date == 'Today':
                year, month, day = today.year, num2abbr[today.month], today.day
            elif date == 'Yesterday':
                yesterday = today - timedelta(days = 1)
                year, month, day = yesterday.year, num2abbr[yesterday.month], (yesterday.day - 1)
            elif bool(re.match('\d{4}', date.split(" ")[-1])):
                month, year = date.split(" ")
                day = '1'
            elif bool(re.match('\d{1,2}', date.split(" ")[-1])):
                month, day = date.split(" ")
                year = today.year
            else:
                print("no date")
                
            date_pub.append(f"{year}-{abbr2num[month]}-{day}")
            commenters.append(name)

    return date_pub, commenters

def get_reviewer_loc_upvotes(soup):
    try:
        locs = []
        contribs = []
        helpful_votes = []
        
        for review in soup.find_all("div", {"class": "MziKN"}):
            if len(review.text) > 0:
                loc, contrib, helpful_vote = re.split("(\d{1,4} contributions?)", review.text)
                helpful_vote = re.sub(" helpful votes?", "", helpful_vote)
                contrib = re.sub(" contributions?", "", contrib)
                
                locs.append(loc)
                contribs.append(contrib)
                helpful_votes.append(helpful_vote)
    
    # not impossible, I guess
    except:
        nb_reviews = len(soup.find_all("q", {"class": "QewHA H4 _a"}))
        locs = ['' for _ in range(nb_reviews)]
        contribs = ['' for _ in range(nb_reviews)]
        helpful_votes = ['' for _ in range(nb_reviews)]
    
    return locs, contribs, helpful_votes

def get_flight_info(soup):
    routes = []
    type_f = []
    class_f = []

    # Must manually split because sometimes all the information is not there.
    # That way, we keep the missing information as empty string
    for review in soup.find_all("div", {"class": "IkECb f O"}):
        route = str(review).split("</div>")
        route = [
            re.sub("(<div class.+?>|<span class.+?>|</span>)", "", _) for _ in route
        ]
        route = [r for r in route if len(r) != 0]

        # not elegant but it works
        try:
            routes.append(route[0])
        except:
            routes.append("")

        try:
            type_f.append(route[1])
        except:
            type_f.append("")

        try:
            class_f.append(route[2])
        except:
            class_f.append("")

    return routes, type_f, class_f

def extract_and_write_info(driver, out):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    # get data
    reviews = [_.text for _ in soup.find_all("q", {"class": "QewHA H4 _a"})]
    nb_reviews = len(reviews)
    titles = [re.sub("\n", "", _.text) for _ in soup.find_all("div", {"class": "_a"})]
    routes, type_f, class_f = get_flight_info(soup)
    re_bubbles = '(<div.+?ui_bubble_rating bubble_|0"></span></div>)'
    ratings = [
        int(re.sub(re_bubbles, "", str(_)))
        for _ in soup.find_all("div", {"class": "Hlmiy F1"})
    ]
    
    date_pub, commenters = get_reviewer_name_date(soup)

    locs, contribs, helpful_votes = get_reviewer_loc_upvotes(soup)

    # writing to disk
    vars = [reviews, titles, date_pub, commenters, routes, type_f, class_f, ratings, locs, contribs, helpful_votes ]
    # varnames = ["reviews","titles","date_pub","commenters","routes","type_f","class_f","ratings","loc", "contribs", "helpful_votes"]
    
    len_vars = [len(_) for _ in vars]

    if all([len_vars[i] == len_vars[-1] for i in range(len(vars))]) is False:
        print(f"Page {i}th ({driver.current_url}) had unequal cats")
    
    out_dat = []
    for i in range(nb_reviews):
        out_dat.append([_[i] for _ in vars])

    with open(out, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for line in out_dat:
            writer.writerow(line)

def get_done_data(path):
    current_dat = []
    with open(path, newline='\n') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            current_dat.append(row)
    return current_dat


def main():
    driver = webdriver.Firefox()
    i = 1
    # url = 'https://www.tripadvisor.ca/Airline_Review-d8729156-Reviews-Southwest-Airlines'
    # o = Path('..') / 'data' / 'raw_dat'
    
    airline_name = re.sub("^.+?Reviews-", "", args.url)
    
    driver.get(args.url)
    sleep(1) # let the page load a little bit
    

    # ------------------------------ Choose language ----------------------------- #

    # The element is always there, as far as we know. 1st child is all languages.
    if args.first_lang:
        lang_button = driver.find_element(By.CSS_SELECTOR, "li.ui_radio:nth-child(2) > label:nth-child(2)")
    elif args.second_lang:
        lang_button = driver.find_element(By.CSS_SELECTOR, "li.ui_radio:nth-child(3) > label:nth-child(2)")
    # elif args.third_lang:
    #     lang_button = driver.find_element(By.CSS_SELECTOR, "li.ui_radio:nth-child(4) > label:nth-child(2)")
    else:
        ValueError('We only do the first 3 languages.')

    lang_button.click()
    # lang = 'English'
    lang = re.sub("\(.+\)", "", lang_button.text)


    # --------------- Check whether we already started this airline -------------- #

    output_file = args.o / f'{airline_name}_{lang}.tsv'
    if output_file.exists() and lang == 'English':
        # for some reasons, we can only go back to English
        current_dat = get_done_data(output_file)

        driver.find_element(By.CSS_SELECTOR, "a.ui_button:nth-child(2)").click()
        
        current_page = int((len(current_dat) - len(current_dat) % 5) + 5)
        next_url = re.sub("or5", f"or{current_page}", driver.current_url)
        
        driver.get(next_url)

    # ------------------------------- make scraping ------------------------------ #

    while True:
        try:
            try:
                # show all comments (seems like span can change)
                driver.find_element(By.CSS_SELECTOR, "span.Ignyf._S.Z").click()
            except:
                pass
            
            print(f"Currently doing {i}th page of {airline_name} ({lang})")

            # extract and append info into a tsv file
            extract_and_write_info(driver, output_file)

            # try clicking next
            driver.find_element(By.CSS_SELECTOR, "a.ui_button:nth-child(2)").click()

            i += 1
            sleep(uniform(2.5, 3.5)) # in case tripadvisors look for bots it helps to throw rdmness

        except NoSuchElementException as e:
            print(e)
            print("done")
            driver.quit()
            break


if __name__ == "__main__":
    today = date.today()
    num2abbr = { index: month for index, month in enumerate(calendar.month_abbr) if month }
    abbr2num = { month:index for index, month in enumerate(calendar.month_abbr) if month }

    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--first_lang", action='store_true')
    parser.add_argument("--second_lang", dest='first_lang', action='store_false')
    # parser.add_argument("--third_lang", dest='first_lang', action='store_false')
    parser.set_defaults(first_lang=True)
    parser.add_argument("-o", type=pathlib.Path)
    args = parser.parse_args()

    main()


# import duckdb
# import pandas as pd
# con = duckdb.connect(database=':memory:')
# con.execute("CREATE TABLE complaints(id VARCHAR PRIMARY KEY, reviews VARCHAR, titles VARCHAR, date_pub DATE, commenters VARCHAR, routes VARCHAR, type_f VARCHAR, class_f VARCHAR, ratings INTEGER, loc VARCHAR, contrib VARCHAR, helpful_votes VARCHAR)")

# con.execute("DROP TABLE complaints;")
# DAT_DIR = '..' / ROOT_DIR / 'data'

# df['uniq_ID'] = df['date_pub'].dt.month_name() + df['date_pub'].dt.year.astype(str) + df['commenters'].str.replace(" ", "") + df['titles'].map(lambda x: ''.join(re.split(" ", x)[0:3]))
# df['uniq_ID'] = pd.util.hash_array(df['uniq_ID'].values)

# df = df[~df.uniq_ID.duplicated()]

# for idx, row in df.iterrows():
#     con.execute(
#         "INSERT INTO complaints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
#         [row['uniq_ID'], row['reviews'], row['titles'], row['date_pub'].date(), 
#          row['commenters'], row['routes'], row['type_f'], 
#          row['class_f'], row['ratings'], row['loc']]
#     )
   