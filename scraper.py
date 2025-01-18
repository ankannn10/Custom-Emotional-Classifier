import re
import json
import time
from itertools import islice
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from webdriver_manager.chrome import ChromeDriverManager

def setup_driver():
    """
    Sets up the Chrome WebDriver with necessary options.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--mute-audio")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scrape_title(driver):
    """
    Scrapes the video title from the YouTube page.
    """
    try:
        title_element = driver.find_element(By.CSS_SELECTOR, '#title yt-formatted-string')
        return title_element.text
    except Exception as e:
        print("Error scraping title:", e)
        return None

def scrape_comment_count(driver):
    """
    Scrapes the comment count from the YouTube page.
    """
    try:
        driver.execute_script("window.scrollBy(0, 600);")
        time.sleep(2)
        comment_count_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//yt-formatted-string[contains(@class, "count-text")]'))
        )
        comment_count_text = comment_count_element.text
        return int(re.sub('[^0-9]', '', comment_count_text))
    except Exception as e:
        print("Error scraping comment count:", e)
        return None

def expand_description_box(driver):
    """
    Expands the description box on the YouTube page if it's collapsible.
    """
    try:
        show_more = driver.find_element(By.CSS_SELECTOR, 'tp-yt-paper-button#expand')
        driver.execute_script("arguments[0].click();", show_more)
        time.sleep(2)
    except Exception:
        print("No 'Show more' button found or already expanded.")

def scrape_views_date_tags(driver):
    """
    Scrapes the views, upload date, and tags from the YouTube page.
    """
    try:
        info_element = driver.find_element(By.CSS_SELECTOR, 'yt-formatted-string#info')
        views_match = re.search(r'([\d,\.]+)\sviews', info_element.text)
        views_count = int(views_match.group(1).replace(',', '')) if views_match else None
        date_match = re.search(r'(\d{1,2}\s\w+\s\d{4}|\w+\s\d{1,2},\s\d{4})', info_element.text)
        upload_date = date_match.group(1) if date_match else None
        tags = [tag.text.lstrip('#') for tag in info_element.find_elements(By.CSS_SELECTOR, 'a.yt-simple-endpoint')]
        if not tags:
            tags = ["No tags found"]
        return views_count, upload_date, tags
    except Exception as e:
        print("Error scraping views, upload date, or tags:", e)
        return None, None, []

def scrape_likes(driver):
    """
    Scrapes the number of likes from the YouTube page.
    """
    try:
        like_button = driver.find_element(By.CSS_SELECTOR, 'button[aria-label*="like this video"]')
        likes_text = like_button.get_attribute("aria-label")
        likes_match = re.search(r'([\d,\.]+)', likes_text)
        return int(likes_match.group(1).replace(',', '')) if likes_match else None
    except Exception as e:
        print("Error scraping likes:", e)
        return None

def scrape_channel_info(driver):
    """
    Scrapes channel name and subscriber count from the YouTube page.
    """
    try:
        channel_name_element = driver.find_element(By.CSS_SELECTOR, '#channel-name #text')
        channel_name = channel_name_element.text.strip()
        subscriber_count_element = driver.find_element(By.CSS_SELECTOR, '#owner-sub-count')
        subscriber_count_text = subscriber_count_element.text.strip().replace("subscribers", "").strip()
        if 'K' in subscriber_count_text:
            subscriber_count = int(float(subscriber_count_text.replace('K', '')) * 1000)
        elif 'M' in subscriber_count_text:
            subscriber_count = int(float(subscriber_count_text.replace('M', '')) * 1_000_000)
        else:
            subscriber_count = int(subscriber_count_text.replace(',', ''))
        return channel_name, subscriber_count
    except Exception as e:
        print("Error scraping channel info:", e)
        return None, None

def scrape_transcript(driver):
    """
    Scrapes the transcript from the YouTube page if available.
    """
    try:
        transcript_button = driver.find_element(By.XPATH, '//*[@id="primary-button"]/ytd-button-renderer/yt-button-shape/button')
        transcript_button.click()
        time.sleep(2)

        transcript = ""
        index = 1
        while True:
            try:
                segment_xpath = f'//*[@id="segments-container"]/ytd-transcript-segment-renderer[{index}]/div/yt-formatted-string'
                text_element = driver.find_element(By.XPATH, segment_xpath)
                transcript += text_element.text + "\n"
                index += 1
            except:
                break

        return transcript
    except Exception as e:
        print("Error scraping transcript:", e)
        return None

class ScraperComments:
    def __init__(self, url, sort_by=SORT_BY_POPULAR, comment_limit=20):
        """
        Initializes the ScraperComments class with the video URL, sorting method, and comment limit.
        """
        self.url = url
        self.sort_by = sort_by
        self.comment_limit = comment_limit
        self.comments = []

    def get_comments(self):
        """
        Fetches comments using the YoutubeCommentDownloader library.
        """
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(self.url, sort_by=self.sort_by)
        self.comments = [comment['text'] for comment in islice(comments, self.comment_limit)]

    def scrape_and_save_comments(self):
        """
        Scrapes comments and returns them as a list.
        """
        self.get_comments()
        return self.comments

def save_data_to_json(data, filename="data.json"):
    """
    Saves the scraped data to a JSON file.
    """
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    driver = setup_driver()
    url = "https://youtu.be/sLTuIr7MRdw?si=DwEmtfm6ydxod35N"
    driver.get(url)
    time.sleep(5)

    data = {
        "title": scrape_title(driver),
        "comment_count": scrape_comment_count(driver),
    }
    expand_description_box(driver)
    views, upload_date, tags = scrape_views_date_tags(driver)
    data.update({
        "views": views,
        "upload_date": upload_date,
        "tags": tags,
        "likes": scrape_likes(driver),
        "channel_name": None,
        "subscriber_count": None,
        "transcript": scrape_transcript(driver),
    })
    data["channel_name"], data["subscriber_count"] = scrape_channel_info(driver)
    
    driver.quit()

    scraper = ScraperComments(url)
    data["comments"] = scraper.scrape_and_save_comments()
    
    save_data_to_json(data)

def scrape_video_data(url, driver):
    """
    Scrapes all video data given a URL.
    """
    try:
        driver.get(url)
        time.sleep(5)

        data = {
            "url": url,
            "title": scrape_title(driver),
            "comment_count": scrape_comment_count(driver),
        }
        expand_description_box(driver)
        views, upload_date, tags = scrape_views_date_tags(driver)
        data.update({
            "views": views,
            "upload_date": upload_date,
            "tags": tags,
            "likes": scrape_likes(driver),
            "channel_name": None,
            "subscriber_count": None,
            "transcript": scrape_transcript(driver),
        })
        data["channel_name"], data["subscriber_count"] = scrape_channel_info(driver)
        
        scraper = ScraperComments(url)
        data["comments"] = scraper.scrape_and_save_comments()

        return data

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
