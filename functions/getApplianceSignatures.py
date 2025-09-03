# Cloud Function to retrieve all appliance signatures for a user from Firestore.

from firebase_functions import https_fn
from firebase_admin import firestore
import firebase_admin

# Initialize Firebase Admin SDK if it hasn't been initialized already.
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.client()

@https_fn.on_call()
def getApplianceSignatures(req: https_fn.CallableRequest):
    """
    Retrieves all appliance signatures stored in Firestore for the authenticated user.

    Args:
        req (https_fn.CallableRequest): The request object containing the authenticated
                                       user's ID and app ID.
    Returns:
        A list of appliance signature documents.
    """
    print("Received request to get appliance signatures.")

    # Validate the user is authenticated. This function requires authentication.
    if req.auth is None or req.auth.uid is None:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.UNAUTHENTICATED,
            message="This function requires authentication."
        )

    user_id = req.auth.uid
    app_id = req.auth.token.get("app_id")
    
    if not app_id:
        print("Warning: app_id not found in auth token. Using a placeholder.")
        app_id = "default-app-id"

    # Construct the path to the private collection for this user.
    # This path must match the security rules you have defined.
    collection_path = f"artifacts/{app_id}/users/{user_id}/appliance_signatures"

    try:
        # Get a reference to the collection.
        collection_ref = db.collection(collection_path)

        # Get all documents in the collection.
        docs = [doc.to_dict() for doc in collection_ref.stream()]

        print(f"Found {len(docs)} signatures for user {user_id}.")

        # Return the list of documents.
        return docs

    except Exception as e:
        print(f"Error fetching appliance signatures: {e}")
        # Return a user-friendly error message.
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=f"An error occurred while fetching data: {e}"
        )
