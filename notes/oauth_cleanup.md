# OAuth Credentials Cleanup

After you are done uploading results to Google Drive, follow these steps to remove all OAuth-related artifacts.

## 1. Local files

Delete the cached credentials and client secrets from the project directory:

```bash
rm /home/lukau/Desktop/point2cad_repr/client_secrets.json
rm /home/lukau/Desktop/point2cad_repr/credentials.json
```

## 2. Revoke app access from your Google account

1. Go to https://myaccount.google.com/permissions
2. Find the app (it will appear under the project name, likely "node-red" from the old GCP project)
3. Click it and select **Remove Access**

This ensures the OAuth token can no longer be used even if someone obtained `credentials.json`.

## 3. Delete the OAuth Client ID from Google Cloud Console

1. Go to https://console.cloud.google.com/apis/credentials
2. Find the OAuth 2.0 Client ID you created (Desktop app)
3. Click the delete icon (trash can) on the right
4. Confirm deletion

## 4. Disable the Google Drive API (optional)

If you no longer need the Drive API in this project:

1. Go to https://console.cloud.google.com/apis/api/drive.googleapis.com/overview
2. Click **Disable API**

## 5. Delete the entire GCP project (nuclear option)

If the GCP project has no other use (it was originally for a node-red experiment):

1. Go to https://console.cloud.google.com/iam-admin/settings
2. Click **Shut down** at the top
3. Enter the project ID to confirm

This removes everything: APIs, credentials, billing, all resources. Google retains the project for 30 days before permanent deletion, during which you can restore it.
