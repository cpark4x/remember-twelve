# Feature 1.5: Google Photos Integration

### Epic Context

**Parent Epic:** [Epic 1: AI-Powered Photo Curation Engine](../../Epics/epic-1-ai-curation-engine.md)

**Epic Objective:** Build an AI-powered curation engine that automatically selects twelve meaningful photos per year for each circle using quality, emotion, diversity, and significance signals.

---

### Feature Overview

**What:** Direct integration with Google Photos API that allows Remember Twelve to authenticate, browse, and curate photos directly from a user's Google Photos library without manual exports.

**Why:**
- **User Benefit**: Seamless access to their entire photo library stored in Google Photos; no manual downloading or organizing required
- **Business Value**: Removes the biggest friction point for users who store photos in the cloud; enables real-time curation of cloud-based libraries
- **Technical Foundation**: Establishes cloud photo provider pattern that can be extended to iCloud, Dropbox, etc.

**Success Criteria:**
- Users can authenticate with Google Photos in <30 seconds
- System can fetch and analyze 1000+ photos from Google Photos library
- Curated photos maintain links back to original Google Photos locations
- Zero data stored permanently (respects user privacy)

---

### Detailed Requirements

#### Authentication Flow

1. **OAuth 2.0 Authentication**
   - Use Google Photos API OAuth flow
   - Request minimal required scopes: `photoslibrary.readonly`
   - Store credentials securely (token encryption)
   - Handle token refresh automatically
   - Support token revocation

2. **User Experience**
   - Simple "Connect Google Photos" button
   - Browser-based OAuth flow
   - Clear permission explanations
   - Success confirmation with library stats

#### Photo Library Access

1. **Library Browsing**
   - Fetch photos filtered by date range (year)
   - Support pagination for large libraries (1000+ photos)
   - Read EXIF metadata from Google Photos
   - Access Google Photos face tags (if available)
   - Filter by album (optional)

2. **Metadata Extraction**
   - Photo timestamp (creation date)
   - Location data (if available)
   - Google Photos categories/labels
   - Face detection results from Google Photos
   - Original filename and dimensions

#### Photo Download & Caching

1. **Temporary Download System**
   - Download photos to temporary cache for analysis
   - Support progressive download (analyze as photos arrive)
   - Automatic cleanup after curation complete
   - Respect rate limits (10,000 requests/day)

2. **Caching Strategy**
   - Cache downloaded photos during session
   - Store Google Photos IDs for future reference
   - Never persist user photos permanently
   - Clear cache on app exit or user request

#### Integration with Curation Pipeline

1. **Photo Source Abstraction**
   - Extend existing LibraryScanner to support cloud sources
   - Common interface: LocalPhotoSource, GooglePhotosSource
   - Transparent to existing analyzers (quality, emotional)
   - Seamless integration with TwelveCurator

2. **Results Export**
   - Store Google Photos URLs in curation results
   - Allow users to view selected photos in Google Photos
   - Support re-running curation without re-download
   - Export curated collection as Google Photos album (future)

---

### Technical Architecture

#### Components

1. **GooglePhotosClient**
   - Handles OAuth authentication
   - API request/response handling
   - Token management and refresh
   - Rate limiting and error handling

2. **GooglePhotosSource**
   - Implements PhotoSource interface
   - Fetches photos by date range
   - Downloads photos to temp cache
   - Provides photo metadata

3. **PhotoSourceFactory**
   - Creates appropriate source (local vs Google Photos)
   - Configuration management
   - Source detection and validation

4. **TokenManager**
   - Secure credential storage
   - Token encryption/decryption
   - Automatic refresh
   - Revocation handling

#### API Endpoints Used

- `photoslibrary.mediaItems.list` - List photos
- `photoslibrary.mediaItems.get` - Get photo details
- `photoslibrary.mediaItems.search` - Search by date/filters
- Photo content download endpoint (baseUrl)

#### Data Flow

```
User → OAuth Flow → GooglePhotosClient → Google Photos API
                                       ↓
                         Fetch Photos (filtered by year)
                                       ↓
                         Download to temp cache
                                       ↓
                    PhotoQualityAnalyzer + EmotionalAnalyzer
                                       ↓
                              TwelveCurator
                                       ↓
                    Results (with Google Photos links)
                                       ↓
                              Web Viewer
```

---

### User Stories

#### Story 1: First-Time Google Photos Connection
**As a** Remember Twelve user with photos in Google Photos
**I want to** connect my Google Photos account
**So that** I can curate my photos without downloading them manually

**Acceptance Criteria:**
- [ ] "Connect Google Photos" button visible on main screen
- [ ] Clicking button opens Google OAuth consent screen
- [ ] After approval, confirmation shows "Connected: [email]"
- [ ] System displays library stats (total photos, date range)
- [ ] Connection status persists across app sessions

#### Story 2: Curate Year from Google Photos
**As a** connected Google Photos user
**I want to** curate photos from a specific year in my Google Photos library
**So that** I can see my best memories without local storage

**Acceptance Criteria:**
- [ ] Can select year to curate (2020-2024, etc.)
- [ ] System fetches all photos from that year
- [ ] Progress indicator shows download/analysis status
- [ ] Curation completes with 12 selected photos
- [ ] Results show thumbnails and Google Photos links
- [ ] Can click photo to open in Google Photos web/app

#### Story 3: Privacy and Data Cleanup
**As a** privacy-conscious user
**I want to** ensure my photos aren't stored permanently
**So that** I feel safe using the service

**Acceptance Criteria:**
- [ ] Clear privacy message: "Photos downloaded temporarily for analysis only"
- [ ] Cache automatically cleaned after curation
- [ ] "Clear Cache Now" button available
- [ ] "Disconnect Google Photos" removes all credentials
- [ ] No photo data persists after disconnect

---

### Implementation Phases

#### Phase 1: Authentication & Basic Fetching (MVP)
- Google OAuth 2.0 setup
- GooglePhotosClient with basic auth
- Fetch photos from library
- Display connection status

**Estimated Effort:** 2-3 days
**Dependencies:** Google Cloud project setup, API credentials

#### Phase 2: Download & Caching
- Temporary photo download system
- Cache management
- Integration with existing analyzers
- Progress tracking

**Estimated Effort:** 2-3 days
**Dependencies:** Phase 1 complete

#### Phase 3: Full Curation Integration
- PhotoSource abstraction layer
- GooglePhotosSource implementation
- End-to-end curation pipeline
- Results with Google Photos links

**Estimated Effort:** 2-3 days
**Dependencies:** Phase 2 complete

#### Phase 4: Enhanced Features (Future)
- Album-based curation
- Export to Google Photos album
- Shared album support
- Face tag integration

**Estimated Effort:** 3-5 days
**Dependencies:** Phase 3 complete, user feedback

---

### Dependencies

**External:**
- Google Cloud project with Photos API enabled
- OAuth 2.0 credentials (Client ID, Secret)
- Google Photos API Python library

**Internal:**
- Feature 1.1: Photo Quality Analyzer ✅
- Feature 1.2: Emotional Significance Detector ✅
- Feature 1.4: Twelve Curation Engine ✅

---

### Success Metrics

**Primary:**
- **Google Photos Adoption Rate**: % of users who connect Google Photos
- **Curation Completion Rate**: % of Google Photos curations that complete successfully
- **Time to First Curation**: Average time from connect to first result

**Secondary:**
- **Photo Fetch Success Rate**: % of photos successfully downloaded
- **API Error Rate**: % of API calls that fail
- **Average Library Size**: Number of photos per user

**Anti-Metrics:**
- Cache storage usage (should stay minimal)
- Credential security incidents (should be zero)

---

### Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Google API rate limits hit | High | Medium | Implement exponential backoff, batch requests efficiently |
| OAuth token expiry during long curation | Medium | Medium | Implement automatic token refresh, handle mid-process refresh |
| Large libraries timeout (10K+ photos) | High | Medium | Implement streaming download, show progress, allow cancellation |
| Photo format compatibility (HEIC, RAW) | Medium | Low | Convert unsupported formats, gracefully skip problematic files |
| User revokes access mid-curation | Low | Low | Graceful error handling, clear user messaging |

---

### Future Enhancements

1. **Multi-Provider Support**
   - iCloud Photos integration
   - Dropbox/OneDrive support
   - Instagram import

2. **Advanced Features**
   - Create curated album in Google Photos
   - Share curated collection
   - Collaborate on family curation

3. **Performance Optimizations**
   - Parallel photo downloads
   - Incremental curation (analyze new photos only)
   - Smart caching based on usage patterns

---

### References

- [Google Photos API Documentation](https://developers.google.com/photos)
- [OAuth 2.0 for Mobile & Desktop Apps](https://developers.google.com/identity/protocols/oauth2/native-app)
- [Python Google Photos Library](https://github.com/googleapis/google-api-python-client)

---

**Status:** 📋 Specification Complete
**Next Step:** Design authentication flow and API architecture
**Owner:** Development Team
**Created:** 2025-10-14
