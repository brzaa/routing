# 🚀 Deployment Guide - Streamlit Web App

This guide explains how to deploy the Route Optimization System as a web application using Streamlit Cloud (FREE).

## 📱 Live Web App Features

- **📤 CSV Upload** - Drag & drop your delivery data
- **⚙️ Interactive Configuration** - Adjust clustering and routing parameters
- **🗺️ Real-time Visualization** - See before/after clustering maps
- **📊 Business Metrics** - View cost savings, distance reduction, CO2 impact
- **📥 Export Results** - Download optimized routes and metrics

---

## 🎯 Option 1: Streamlit Cloud (Recommended - FREE)

### Prerequisites
- GitHub account
- This repository pushed to GitHub

### Deployment Steps

#### 1. **Push Code to GitHub** (Already Done ✅)
Your repository is at: `https://github.com/brzaa/routing`

#### 2. **Sign Up for Streamlit Cloud**
- Go to: https://streamlit.io/cloud
- Click "Sign up" and use your GitHub account
- It's completely FREE!

#### 3. **Deploy Your App**

**Option A: Direct Link**
1. Go to: https://share.streamlit.io/deploy
2. Fill in the form:
   - **Repository**: `brzaa/routing`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
3. Click **"Deploy!"**

**Option B: From Streamlit Cloud Dashboard**
1. Log in to https://share.streamlit.io/
2. Click **"New app"**
3. Select your repository: `brzaa/routing`
4. Branch: `main`
5. Main file: `streamlit_app.py`
6. Click **"Deploy!"**

#### 4. **Wait for Deployment** (2-5 minutes)
Streamlit Cloud will:
- Clone your repository
- Install dependencies from `requirements.txt`
- Start your app
- Give you a public URL like: `https://brzaa-routing-app.streamlit.app`

#### 5. **Share Your App** 🎉
Your app will be live at a URL like:
```
https://[your-username]-routing-[random-id].streamlit.app
```

Share this link with anyone - no login required for users!

### Managing Your App

**View Logs:**
- Click on your app in the Streamlit Cloud dashboard
- View real-time logs and performance

**Update App:**
- Just push to GitHub `main` branch
- Streamlit Cloud auto-deploys changes
- Updates appear within 1-2 minutes

**App Settings:**
- Configure secrets (if needed)
- Set custom domain (paid feature)
- Manage resources

---

## 🎯 Option 2: Local Development

### Run Locally

```bash
# Clone repository
git clone https://github.com/brzaa/routing.git
cd routing

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

The app will open at: `http://localhost:8501`

### Local Testing Tips
- Use smaller datasets first (< 500 deliveries)
- Reduce time limit for faster testing
- Check browser console for errors

---

## 🎯 Option 3: Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t routing-app .

# Run container
docker run -p 8501:8501 routing-app
```

### Deploy to Cloud Platforms

**Railway.app:**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
railway up
```

**Render.com:**
1. Connect GitHub repository
2. Select "Docker" deployment
3. Render auto-detects Dockerfile
4. Deploy!

---

## 🎯 Option 4: Vercel/Netlify (NOT Recommended)

⚠️ **Note**: Vercel and Netlify are designed for static sites and serverless functions. They don't support long-running Python processes like Streamlit well.

**Better alternatives for Python apps:**
- ✅ Streamlit Cloud (best for Streamlit apps)
- ✅ Railway.app (easy Python deployment)
- ✅ Render.com (free tier available)
- ✅ Heroku (paid, but reliable)

---

## 📊 Performance & Limits

### Streamlit Cloud Free Tier
- **Resources**: 1 GB RAM, 1 CPU core
- **Storage**: 50 GB
- **Apps**: Unlimited public apps
- **Bandwidth**: Generous limits
- **Uptime**: Apps sleep after inactivity (wake up in ~30 seconds)

### Optimization Tips

**For Large Datasets (> 1000 deliveries):**
1. Reduce `time_limit` to 20 seconds
2. Increase `max_packages_per_pod` to create fewer clusters
3. Process in batches if needed

**Cache Results:**
```python
@st.cache_data
def load_data(file):
    return pd.read_csv(file)
```

**Memory Management:**
- Streamlit Cloud has 1 GB RAM limit
- Large datasets (> 5000 deliveries) may hit limits
- Consider preprocessing data before upload

---

## 🔒 Security Considerations

### Data Privacy
- Uploaded data is processed in memory only
- Data is NOT stored permanently
- Each session is isolated
- Use `.gitignore` to prevent committing sensitive data

### Best Practices
- Don't upload files with sensitive personal information
- Consider encrypting sensitive columns before upload
- Use environment variables for API keys (if needed)

### Add Secrets (if needed)
In Streamlit Cloud dashboard:
1. Go to app settings
2. Click "Secrets"
3. Add key-value pairs
4. Access in code: `st.secrets["key"]`

---

## 🐛 Troubleshooting

### Common Issues

**1. "Module not found" error**
- Check `requirements.txt` has all dependencies
- Redeploy app from Streamlit Cloud dashboard

**2. App crashes on large files**
- Reduce dataset size
- Increase `maxUploadSize` in `.streamlit/config.toml`
- Reduce time limit for optimization

**3. Maps not displaying**
- Check browser console for errors
- Ensure Folium HTML is generated correctly
- Try different browser

**4. Slow performance**
- Reduce `time_limit` parameter
- Use fewer deliveries for testing
- Optimize clustering parameters

### View Logs
In Streamlit Cloud:
1. Click on your app
2. View "Logs" tab
3. Check for errors and warnings

---

## 📝 Custom Domain (Optional - Paid)

Streamlit Cloud Pro plan allows custom domains:
1. Purchase domain (e.g., GoDaddy, Namecheap)
2. Upgrade to Streamlit Cloud Pro
3. Add CNAME record pointing to Streamlit
4. Configure in app settings

---

## 🎉 Success Checklist

After deployment, verify:
- [ ] App loads without errors
- [ ] Can upload CSV file
- [ ] Clustering runs successfully
- [ ] Maps display correctly
- [ ] Metrics show accurate results
- [ ] Can download exports
- [ ] Share URL works for others

---

## 📚 Additional Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Streamlit Cloud**: https://streamlit.io/cloud
- **Community Forum**: https://discuss.streamlit.io
- **GitHub Repository**: https://github.com/brzaa/routing

---

## 🆘 Support

Need help?
1. Check Streamlit Community Forum
2. Open issue on GitHub: https://github.com/brzaa/routing/issues
3. Review error logs in Streamlit Cloud dashboard

---

**Built with ❤️ using Streamlit, OR-Tools, and scikit-learn**
