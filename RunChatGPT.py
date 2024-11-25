import os  
from openai import AzureOpenAI 
from dotenv import load_dotenv
#from azure.identity import DefaultAzureCredential, get_bearer_token_provider  


def main():
    try:

        show_citations = False 

        load_dotenv()
        endpoint = os.getenv("OpenAI_ENDPOINT_URL")  
        deployment = os.getenv("AZURE_OAI_DEPLOYMENT_NAME")
        key = os.getenv("OpenAI_KEY")
         
        #Initialize the Azure OpenAI client        
        client = AzureOpenAI(
            azure_endpoint= f"{endpoint}", #you can add this to the url for RAG capabilities /openai/deployments/{deployment}/extensions",
            api_key= key,  
            #azure_ad_token_provider=token_provider,  
            api_version= "2024-10-21",  
        )
        
        
        # Get the prompt
        text = input('\nEnter a question:\n')

        # Send request to Azure OpenAI model
        print("...Sending the following request to Azure OpenAI endpoint...")
        print("Request: " + text + "\n")  
                
        completion = client.chat.completions.create(  
            model=deployment,  
            messages= [
                {"role": "system", "content": "You are an AI assistant that helps people find information."},
                {"role": "user", "content": text}
                ],  
            max_tokens = 800,  
            temperature = 0.7,  
        )

        # Print response
        print("Response: " + completion.choices[0].message.content + "\n")  
                
         


    except Exception as ex:
        print(ex) 

if __name__ == '__main__':
    main()