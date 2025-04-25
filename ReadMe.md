# AutoQuest
## Prerequisites
- Python 3.8 or higher
- Docker Desktop
- WSL 2

## Procedure
### Setup the Vector Database
> Below steps are for windows, for other os look into the link: [link](https://milvus.io/docs/prerequisite-docker.md)
- Once you have your environment setup, you need to download the installation script for Milvus Vector DB environment.
> Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat 
- Run the script to set up the Milvus environment by executing `standalone.bat` in your command line.
> ./standalone.bat start
- If you see the below message in the terminal your DB is up and running successfully.
> Wait for Milvus starting...\
> Start successfully.\
> To change the default Milvus configuration, edit user.yaml and restart the service.
- For more info look into the website: [Install Milvus Standalone](https://milvus.io/docs/install_standalone-windows.md)

### Install the application requirements
- You can find the requirements.txt file with the application code. Run the below command to install the requirements. The *requirements.txt* is in f1 folder.
> pip install -r requirements.txt
- If you have no errors proceed, else fix the errors

### Download the required models
- Download the model at [link](https://drive.google.com/file/d/1jEceRDWFSn_kuUrBQYGyaJE09LwZ7eAI/view?usp=drive_link) and rename to *model1_t5.pt*.
- Save this file in the model folder of the given code.

### Run the code
- To start the frontend of the application, you need to run the *app.py* file.
> streamlit run app.py
- Open your web browser and navigate to `http://localhost:8501` to access the application.
- Ensure that the Milvus server is running before accessing the frontend. If you encounter any issues, check the logs for troubleshooting.



## Resources
1. Model 1 (T5): [link](https://drive.google.com/file/d/1jEceRDWFSn_kuUrBQYGyaJE09LwZ7eAI/view?usp=drive_link)
2. Model 2 (Bert-GPT2): [link](https://drive.google.com/file/d/1d7kbr9NKNJjxxIBN1oAazyB0LliVNxBE/view?usp=sharing )
3. Datasets: [link](https://github.com/arjunkarthikeyanakka/AutoQuest/tree/main/datasets)
4. Github Code Repo: [link](https://github.com/arjunkarthikeyanakka/AutoQuest)
5. Code Execution Video: [link](https://www.loom.com/share/7f3430148d404d17b7906aeb877f5cf9)