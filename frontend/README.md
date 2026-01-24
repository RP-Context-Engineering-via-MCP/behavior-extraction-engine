# Behavior Extraction System - UI Demo Guide

## Overview

This frontend provides a visual interface for the Behavior Extraction and Management System, allowing you to:
- Input natural language prompts
- See extracted behaviors in real-time
- Track what happened to each behavior (new, duplicate, conflict, etc.)
- View all stored behaviors and conflicts
- Visualize the complete behavior processing flow

## Features

### 1. **Behavior Extraction Interface**
- Input field for user prompts
- Session ID management
- Real-time extraction with loading indicators

### 2. **Processing Summary**
Visual cards showing:
- 📝 Total behaviors extracted
- 💾 Behaviors stored to database
- 🔁 Behaviors reinforced (duplicates)
- ⚠️ Conflicts detected
- ✂️ Behaviors pruned (low credibility)

### 3. **Detailed Flow Tracking**
Each behavior shows:
- **Action taken**: New, Duplicate, Conflict, Superseded, Ignored, or Pruned
- **Canonical fields**: Intent, Target, Context, Polarity
- **Credibility score**: Initial credibility calculation
- **Matched behavior**: If duplicate or conflict, shows the existing behavior
- **Conflict details**: LLM analysis, resolution type, explanations
- **Semantic distance**: Embedding similarity score

### 4. **Color-Coded Actions**
- 🟢 **Green**: New behaviors or compatible
- 🔵 **Blue**: Duplicates (reinforced)
- 🔴 **Red**: Conflicts detected
- 🟣 **Purple**: Existing behavior superseded
- ⚪ **Gray**: New behavior ignored
- 🟡 **Yellow**: Pruned behaviors

### 5. **View All Data**
- **All Behaviors**: Browse complete behavior database for the session
- **All Conflicts**: Review all detected conflicts with details

## Getting Started

### 1. Start the Backend Server

```bash
# Make sure you're in the project root directory
cd "/home/dilshangamage/My work/Research implementations/Behavior detaction and management"

# Activate your Python environment if needed
# source venv/bin/activate

# Start the FastAPI server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access the Frontend

Open your browser and navigate to:
```
http://localhost:8000/frontend/index.html
```

### 3. Use the Interface

1. **Enter a Session ID** (default: "demo_user")
   - Different sessions maintain separate behavior databases
   
2. **Input your prompt** in the textarea
   - Example: "I prefer dark mode in my IDE. I like using Python for backend development. I don't like verbose code."

3. **Click "Extract Behaviors"**
   - The system will process your prompt
   - Show extraction results with detailed flow information

4. **Review the results**:
   - Check the summary cards for quick stats
   - Scroll through individual behavior flow cards
   - Each card shows what happened to that specific behavior

5. **Explore additional data**:
   - Click "View All Stored Behaviors" to see the complete database
   - Click "View All Conflicts" to review detected conflicts

## Example Prompts

### Test Case 1: New Behaviors
```
I prefer dark mode when coding. I like using TypeScript for frontend development.
```
Expected: Both behaviors will be stored as new entries.

### Test Case 2: Duplicate Detection
First prompt:
```
I prefer dark mode when coding.
```
Second prompt (same session):
```
I really like dark mode in my IDE.
```
Expected: The second will reinforce the first (duplicate detection).

### Test Case 3: Conflict Detection
First prompt:
```
I prefer dark mode for coding.
```
Second prompt (same session):
```
I prefer light mode for coding.
```
Expected: Conflict detected (polarity mismatch on same target).

### Test Case 4: Context-Dependent Behaviors
```
I prefer dark mode in my IDE. I prefer light mode for documentation. I like using Python.
```
Expected: Dark mode and light mode are compatible (different contexts).

### Test Case 5: Low Credibility Pruning
```
Maybe I might possibly like dark mode, I'm not sure.
```
Expected: May be pruned due to low confidence/clarity.

## API Endpoints Used

### POST /extract-detailed
Main extraction endpoint with flow tracking.

**Request:**
```json
{
  "prompt": "Your prompt here",
  "session_id": "demo_user"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "extraction": {
      "extraction_time_ms": 1234.56,
      "total_segments": 1
    },
    "processing": {
      "total_extracted": 3,
      "total_stored": 2,
      "total_reinforced": 1,
      "total_conflicts": 0,
      "total_pruned": 0
    },
    "flow_info": [
      {
        "behavior_description": "...",
        "action": "NEW_BEHAVIOR",
        "credibility": 0.85,
        "canonical": {...},
        "details": "..."
      }
    ]
  }
}
```

### GET /behaviors/{user_id}
Retrieve all behaviors for a user.

### GET /conflicts/{user_id}
Retrieve all conflicts for a user.

## Understanding the Flow

### Action Types

1. **NEW_BEHAVIOR** ✨
   - No matching behavior found
   - Created new database entry
   - Shows stored behavior ID

2. **DUPLICATE_REINFORCED** 🔁
   - Matched existing behavior
   - Same intent, target, context, polarity
   - Existing behavior's credibility and reinforcement count increased

3. **CONFLICT_DETECTED** ⚠️
   - Conflicting behaviors found
   - Both behaviors flagged for user review
   - Shows LLM conflict analysis

4. **CONFLICT_AUTO_RESOLVED** ✅
   - Conflict automatically resolved based on credibility
   - Shows which behavior won and why

5. **SUPERSEDED_EXISTING** 🔄
   - New behavior replaced existing one
   - Existing behavior marked as superseded
   - New behavior stored with higher credibility

6. **IGNORED_NEW** 🚫
   - New behavior ignored
   - Existing behavior has higher credibility
   - Nothing stored

7. **PRUNED** ✂️
   - Behavior filtered out before storage
   - Reasons: low credibility, missing fields, embedding failure

## Troubleshooting

### Frontend not loading?
- Check if the backend server is running on port 8000
- Verify the API_BASE_URL in script.js matches your server address

### CORS errors?
- The backend should allow CORS by default
- If issues persist, add CORS middleware to app.py

### No results showing?
- Check browser console for JavaScript errors
- Verify the API responses in Network tab
- Ensure the database connection is working

### Behaviors not appearing in "View All"?
- Make sure you're using the same session_id
- Check if behaviors were actually stored (not pruned)

## Technical Details

### Technologies Used
- **Frontend**: Pure HTML, CSS, JavaScript (no frameworks)
- **Backend**: FastAPI (Python)
- **Styling**: Custom CSS with modern design patterns
- **API**: RESTful JSON endpoints

### Key Features
- Responsive design (mobile-friendly)
- Real-time updates
- Modal dialogs for detailed views
- Color-coded visual indicators
- Smooth animations and transitions

## Demo Tips

For research evaluation panels:

1. **Start with a simple example** to show basic extraction
2. **Demonstrate duplicate detection** with similar prompts
3. **Show conflict detection** with opposite preferences
4. **Highlight the LLM analysis** in conflict scenarios
5. **Display the canonical reasoning** (intent, target, context, polarity)
6. **Show the credibility-based auto-resolution**
7. **Demonstrate the complete behavior database** view

## Notes

- The UI is designed for demonstration purposes
- All behavior processing logic remains in the backend (unchanged)
- The frontend only visualizes what the backend does
- Session IDs allow testing with different user profiles
- The system maintains full history of all behaviors and conflicts
