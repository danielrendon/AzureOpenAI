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
            api_version= "2023-09-01-preview",  
        )

        #Create  system message
        system_message = "You are a helpful AI assistant."
        #Initialize message array
        messages_array = [{"role": "system", "content": system_message}]
        
        while True:
            # Get the prompt
            text = input("\nEnter your question(or type 'quit' to exit):\n")
            if text.lower() == 'quit':
                break
            if len(text) == 0:
                print("Please enter a prompt. If you want to exit, type 'quit'")
                continue

            # Send request to Azure OpenAI model
            print("...Sending the following request to Azure OpenAI endpoint...")
            print("Request: " + text + "\n") 

            #To keep chat history
            messages_array.append({"role": "user", "content": text})    

            response = client.chat.completions.create(  
                model=deployment,  
                messages= messages_array,  
                max_tokens = 1200,  
                temperature = 0.7,
                top_p=0.95  
            )

            generated_text = response.choices[0].message.content

            #Add generated text to messages array
            messages_array.append({"role": "assistant", "content": generated_text})

            
            # Print response
            print(generated_text)

            #print("Response: " + response.choices[0].message.content + "\n")  
    
    except Exception as ex:
        print(ex) 

if __name__ == '__main__':
    main()