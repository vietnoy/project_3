# GitHub Setup Instructions

Follow these steps to push your project to GitHub so you can pull it on your Mac.

## Step 1: Create GitHub Repository

1. Go to [github.com](https://github.com) and log in
2. Click the **+** button in top-right → **New repository**
3. Fill in:
   - **Repository name**: `smart-contract-security-rag` (or any name you like)
   - **Description**: "Smart Contract Security Assistant using RAG with LangChain"
   - **Visibility**: Public or Private (your choice)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **Create repository**

## Step 2: Push to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
cd "c:\Users\admin\Desktop\project_3"

# Add GitHub as remote
git remote add origin https://github.com/YOUR-USERNAME/smart-contract-security-rag.git

# Push to GitHub
git push -u origin master
```

**Replace `YOUR-USERNAME`** with your actual GitHub username!

### If you use SSH instead of HTTPS:
```bash
git remote add origin git@github.com:YOUR-USERNAME/smart-contract-security-rag.git
git push -u origin master
```

## Step 3: Verify Upload

1. Refresh your GitHub repository page
2. You should see all files uploaded:
   - README.md
   - src/ folder with Python files
   - sample-smart-contract-dataset/ with 1000+ JSON files
   - Documentation files

## Step 4: On Your Mac

Once pushed to GitHub, on your Mac:

```bash
# Open Terminal
cd ~/Desktop  # or wherever you want

# Clone the repository
git clone https://github.com/YOUR-USERNAME/smart-contract-security-rag.git

# Enter the directory
cd smart-contract-security-rag

# Follow GETTING_STARTED.md to set up
```

## Troubleshooting

### Authentication Required

If Git asks for authentication:

**Option 1: Personal Access Token (Recommended)**
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control of private repositories)
4. Copy the token
5. When Git asks for password, paste the token (not your GitHub password!)

**Option 2: GitHub CLI**
```bash
# Install GitHub CLI
winget install GitHub.cli

# Authenticate
gh auth login

# Then push
git push -u origin master
```

### "Remote origin already exists"

If you get this error:
```bash
git remote remove origin
git remote add origin https://github.com/YOUR-USERNAME/smart-contract-security-rag.git
git push -u origin master
```

### Large File Warning

If you get warnings about large files (the dataset is ~6MB total), that's fine. Git will handle it.

If you want to use Git LFS for large files:
```bash
git lfs install
git lfs track "*.json"
git add .gitattributes
git commit -m "Configure Git LFS"
git push
```

## What's Already Set Up

✅ Git repository initialized
✅ All files committed (927 files, 51,305 lines)
✅ .gitignore configured (won't upload .env, venv/, chroma_db/)
✅ Ready to push to GitHub

## Summary

1. Create new GitHub repo (don't initialize with anything)
2. Run: `git remote add origin https://github.com/YOUR-USERNAME/REPO-NAME.git`
3. Run: `git push -u origin master`
4. On Mac: `git clone https://github.com/YOUR-USERNAME/REPO-NAME.git`

That's it! Your project will be synced to GitHub and ready to pull on your Mac.
