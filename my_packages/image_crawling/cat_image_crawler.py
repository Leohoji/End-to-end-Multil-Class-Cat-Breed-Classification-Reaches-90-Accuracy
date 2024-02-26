import os
import glob
import time
import urllib
import shutil 
import random
from pathlib import Path
from urllib.request import urlretrieve
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class CatImageCrawler:
    """
    CatImageCrawler class object can crawl cats images, 
    save images crawled, combine crawled images and original images, and export all the images.

    Attributes:
        service (str): The local address of webdriver.exe for selenium module.
        home_dir (str): Home directory of target folder.
    """
    def __init__(self, service, home_dir):
        self.__service = service
        self.home_dir = home_dir

        if os.path.exists(home_dir):
            self.home_dir = home_dir
            print(f"Set successfully as {self.home_dir}")
        
        # Reassigned during running methods
        self.animal_breeds = ""
        self.model_labels = ""

        # Target address for crawling
        self.__URL = 'https://www.google.com.tw/images'
        
        print("CatImageFileOperator object has built.")
    
    def __create_image_path(self, breed_name:str) -> str:
        """
        Create a new dir if not exists.
    
        Args:
            breed_name (str):  Subdirectory of main folder.
    
        Return:
            Target full path of the home directory and breed_name.
        """
        # Create folder path
        target_path = os.path.join(self.home_dir, breed_name + '_crawl')
    
        # Create folder if not exists
        if not os.path.isdir(target_path): 
            os.mkdir(target_path)
            
        return target_path

    
    def __set_headers(self):
        """
        Set headers before using webdriver server.
        """

        # Set header
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

    
    def _reset_server(self, new_service:str):
        """
        Reset webdriver serveice.

        Args:
            new_service: New service address in local.
        """
        self.__service = new_service
        print(f'Service ({self.__service}) was reset successfully.')
    

    def get_class_names(self):
        """
        Set cat breed-names after setting home directory.
        """
        # Generate class names
        animal_paths = Path(self.home_dir) # get path
        class_names = [p.name for p in animal_paths.glob('*')] # get class names
        print(f"There are {len(class_names)} class names:\n{class_names}")
        
        self.animal_breeds = class_names
        return self.animal_breeds


    def crawling_from_google_images(self, image_number:int=250, data=None):
        """
        Crawl several animal images from chrome server, and save them.
        
        Original folder:
        
        main_directory/
        |
        |___class_a/
        |    |___a_image_1.jpg
        |    |___a_image_2.jpg
        |    |___...
        |
        |___class_b/
             |___b_image_1.jpg
             |___b_image_2.jpg    
        
        Save format:
        
        main_directory/
        |
        |___class_a_crawl/
        |    |___crawl_a_image_1.jpg
        |    |___crawl_a_image_2.jpg
        |    |___...
        |
        |___class_b_crawl/
             |___crawl_b_image_1.jpg
             |___crawl_b_image_2.jpg
        
        Args:
            image_number: The number of images you want to crawl.
        """
        # Set class names
        class_names = data if data else self.animal_breeds


        # Set header
        self.__set_headers()
            
        # Visit web and search images
        for class_name in class_names:

            # cat breed
            breed = class_name
            
            # Set the service - chrome
            try:
                service = Service(executable_path=self.__service)
                driver = webdriver.Chrome(service=service)
                driver.get(self.__URL)
            except Exception as e:
                print(e.__class__.__name__)
                print("Your service is something wrong, please reset it.")
                return
            
            # Find the search bar and search keyword
            keyword = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            keyword.clear()
            print(f"Searching {breed} ...")
            keyword.send_keys('%s cats all images' % (breed))
            keyword.send_keys(Keys.RETURN)
            
            # Collect image src
            image_srcs = []
            target_path = self.__create_image_path(breed)
            
            counts = 1
            while counts < image_number:
                
                # Find images
                images = WebDriverWait(driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".Q4LuWd"))
                )
                for image in images:
                    src = image.get_attribute('src')
        
                    # Check unique path
                    if src in image_srcs:
                        continue
                    image_srcs.append(src)
        
                    # Check whether save the right src
                    if isinstance(src, str):
                        try:
                            save_to = os.path.join(target_path, 'crawl_' + breed + '_' + str(counts) + '.jpg')
                            urlretrieve(src, save_to) # Save images
                            counts += 1
                        except:
                            pass
                print(f"Download {counts} images...")
                
                # Scroll Down Webpage
                driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
                time.sleep(3)
                try:
                    # Input button: "show more results" button
                    more_results = driver.find_element(By.CLASS_NAME, 'LZ4I')

                    # "Looks like you've reached the end" hint
                    end = driver.find_element(By.CLASS_NAME, 'Yu2Dnd').find_element(By.TAG_NAME, 'div').text 

                    # If mouse scroll to bottom, break.
                    if end:
                        print("Looks like you've reached the end !!!!!")
                        break
        
                    more_results.click()
                    time.sleep(3) # wait for loading
                except:
                    pass

            print(f"{breed} Finish, total images: {len(image_srcs)}")
            driver.close()
            
        print("All images have been downloaded, YAA!!!!!!!!")        


    def combine_folders_from_suffix(self, suffix:str='_crawl'):
        """
        Walk home directory and copy files from old folder containing suffix to similar folder without suffix, and remove old folder.
        The folder structure must be include:
                Directory 
        ____________|____________
        |                       |
        sub-directory           sub-directory  + suffix 
                  
        Args:
            src_prefix (str): Specific prefix on old folder.
        """
        
        # Collect directory
        paths = [path for path in Path(self.home_dir).glob(r'*%s'%(suffix))]
        
        # Check directory
        if not paths:
            return "Your suffix does not exist."
        else:
            # Walk directory
            for path in paths:
                # Create target directory
                target_path = os.path.join(self.home_dir, path.name.split('_')[-2])
                print(f"Move files from [{str(path)}] to {str(path.name).split('_')[:-1]}...")
                
                # Walk image file need to be moved
                for jpgfile in glob.glob(os.path.join(str(path), '*.jpg')):
                    shutil.move(jpgfile, target_path)
                print(f"Delete {str(path)}...")
                os.rmdir(str(path))
            print('--------------- Finish ------------------')



__all__ = ['CatImageCrawler']