# Chat UI - Setup and Usage

## 📌 Overview
This project consists of a **React frontend** and a **Flask backend**. You can run the application in two modes:
1. **Development Mode** (Logs enabled, auto-reload)
2. **Production Mode** (Faster, optimized, no logs)

---

## 🚀 Setup Instructions

### 1️⃣ Install Dependencies
Ensure you have **Node.js**, **Python**, and **pip** installed.

```sh
# Install frontend dependencies
cd chat-ui-main
npm install

# Install backend dependencies
pip install -r requirements.txt
```

If using Windows, install `cross-env`:
```sh
npm install cross-env --save-dev
```

---

### 2️⃣ Running the App

#### **🔧 Development Mode (With Logs, Auto-Reload)**
Use this mode if you want **real-time logs** and **auto-reloading** on changes:
```sh
npm run start:dev
```
✅ This will:
- Start the **React frontend**
- Start Flask in **debug mode** (`app.run(debug=True)`) with logs enabled

---

#### **🚀 Production Mode (Optimized, No Logs)**
Use this mode for **faster performance**, but without real-time logs:
```sh
npm run start:prod
```
✅ This will:
- Start the **React frontend**
- Run Flask using **Waitress** (`serve(app, threads=4)`) for better performance

---

### 3️⃣ How the Modes Work
The **`FLASK_ENV`** variable controls the mode:
- **Development (`FLASK_ENV=development`)** → Uses `app.run(debug=True)`
- **Production (`FLASK_ENV=production`)** → Uses `waitress.serve()`

This is set in `package.json`:
```json
"scripts": {
  "start:dev": "cross-env FLASK_ENV=development concurrently \"npm run frontend\" \"npm run backend\"",
  "start:prod": "cross-env FLASK_ENV=production concurrently \"npm run frontend\" \"npm run backend\""
}
```

And read by **`server.py`**:
```python
import os
FLASK_ENV = os.getenv("FLASK_ENV", "production")
if FLASK_ENV == "development":
    app.run(debug=True, port=5000)
else:
    serve(app, host='0.0.0.0', port=5000, threads=4)
```

---

### 4️⃣ Testing
Once the app is running, open your browser:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:5000
- **Real-time logs:** http://localhost:5000/api/logs (only in **development mode**)

To test logs in the terminal:
```sh
curl -N http://localhost:5000/api/logs
```

---

### 5️⃣ Stopping the App
Press `CTRL + C` in the terminal to stop both frontend & backend.

---

## ✅ Summary
| Command               | Mode       | Features                     |
|----------------------|-----------|-----------------------------|
| `npm run start:dev`  | Development | Logs, auto-reload, debug mode |
| `npm run start:prod` | Production  | Faster, optimized, no logs   |

This setup ensures easy switching between **development** and **production** modes. 🚀

