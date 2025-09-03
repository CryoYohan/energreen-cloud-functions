# Cloud Function to update an appliance signature in Firestore.

from firebase_functions import https_fn
from firebase_admin import firestore
import firebase_admin

# Initialize Firebase Admin SDK if it hasn't been initialized already.
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.client()

@https_fn.on_call()
def updateApplianceSignature(req: https_fn.CallableRequest):
    """
    Updates the 'label' field of an appliance signature in Firestore.

    Args:
        req (https_fn.CallableRequest): The request object containing the
                                       authenticated user's ID, app ID,
                                       and the data to be updated.
                                       Data should contain 'signature_id' and 'label'.
    Returns:
        A dictionary with a success message or an error.
    """
    print("Received request to update an appliance signature.")

    # 1. Validate the user is authenticated.
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

    # 2. Get the data from the request.
    try:
        signature_id = req.data["signature_id"]
        label = req.data["label"]
        if not signature_id or not label:
            raise ValueError("signature_id and label are required.")
    except (KeyError, ValueError) as e:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=f"Invalid request format: {e}"
        )

    # 3. Construct the path to the specific document.
    collection_path = f"artifacts/{app_id}/users/{user_id}/appliance_signatures"
    document_ref = db.collection(collection_path).document(signature_id)

    try:
        # 4. Update the document with the new label.
        document_ref.update({"label": label})
        print(f"Signature {signature_id} updated with label '{label}' for user {user_id}.")
        return {"status": "success", "message": "Appliance signature updated successfully."}

    except Exception as e:
        print(f"Error updating appliance signature: {e}")
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=f"An error occurred while updating the data: {e}"
        )
