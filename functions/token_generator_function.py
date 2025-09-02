import functions_framework
import jwt
import datetime
import os
from google.oauth2 import service_account

# This audience claim must be the URL of your final Cloud Function endpoint
# that will be called by the ESP32 client.
AUDIENCE_URL = os.environ.get('RECEIVE_ENERGY_URL')

# This is the full path to your service account key file within the Cloud Function's directory.
# This file is uploaded with the function.
SA_KEY_FILE = 'service_account.json'

@functions_framework.http
def token_generator_function(request):
    """
    HTTP Cloud Function that generates and returns a JWT for device authentication.
    The function requires a service account key file to be deployed with it.
    """
    if request.method != 'GET':
        return ('Method Not Allowed', 405)

    if not AUDIENCE_URL:
        print("Error: RECEIVE_ENERGY_URL environment variable is not set.", file=sys.stderr)
        return ('Server configuration error.', 500)

    try:
        # Load the credentials from the service account key file
        credentials = service_account.Credentials.from_service_account_file(
            SA_KEY_FILE,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )

        # Get the private key from the credentials
        private_key = credentials.private_key
        
        # Define the payload for the JWT
        payload = {
            # iss: The token issuer. In this case, your service account email.
            "iss": credentials.service_account_email,
            # aud: The audience, which is the URL of the receiving Cloud Function.
            "aud": AUDIENCE_URL,
            # iat: Issued at timestamp.
            "iat": datetime.datetime.utcnow().timestamp(),
            # exp: Expiration time. We'll set it to 10 minutes.
            "exp": (datetime.datetime.utcnow() + datetime.timedelta(minutes=10)).timestamp()
        }

        # Encode the JWT using the private key and a secure algorithm.
        signed_token = jwt.encode(
            payload,
            key=private_key,
            algorithm='RS256' # RS256 is the standard algorithm for JWTs with public/private keys
        )

        return (signed_token, 200)

    except FileNotFoundError:
        print(f"Service account key file not found: {SA_KEY_FILE}", file=sys.stderr)
        return ('Server configuration error: Key file missing.', 500)
    except Exception as e:
        print(f"Error generating JWT: {e}", file=sys.stderr)
        return ('Internal Server Error', 500)
