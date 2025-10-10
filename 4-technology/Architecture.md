# Architecture — Remember Twelve

## System Overview

Remember Twelve follows a **mobile-first, cloud-native architecture** with three primary layers:

1. **Client Layer**: Native iOS/Android apps + Progressive Web App
2. **API Layer**: RESTful backend services for user management, photo processing, and curation
3. **AI/ML Layer**: Photo analysis, curation algorithms, and batch processing pipelines

**Architecture Philosophy**:
- **Modular**: Independent services that can evolve separately
- **Scalable**: Handle millions of photos without linear cost increases
- **Durable**: Data formats and storage designed for multi-decade longevity
- **Privacy-First**: User data encrypted, never sold, exportable at any time

---

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                           │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐       │
│  │ iOS App  │  │ Android  │  │ Progressive Web App │       │
│  └──────────┘  └──────────┘  └─────────────────────┘       │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTPS / REST API
┌─────────────────────▼───────────────────────────────────────┐
│                      API LAYER                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ User API    │  │ Photo API    │  │ Circle API   │       │
│  │ (Auth,      │  │ (Upload,     │  │ (CRUD,       │       │
│  │  Profile)   │  │  Metadata)   │  │  Sharing)    │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   AI/ML LAYER                               │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │ Curation Engine │  │ Batch Processor  │                 │
│  │ (Photo Quality, │  │ (Yearly Twelve   │                 │
│  │  Emotion, Dive  │  │  Generation)     │                 │
│  │  rsity)         │  │                  │                 │
│  └─────────────────┘  └──────────────────┘                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   DATA LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ User DB      │  │ Photo Store  │  │ Archive DB   │      │
│  │ (PostgreSQL) │  │ (S3/Blob)    │  │ (Twelves,    │      │
│  │              │  │              │  │  Metadata)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. **Client Layer**

#### iOS Native App
- **Purpose**: Primary mobile experience for iOS users
- **Technology**: Swift, SwiftUI, Combine
- **Interfaces**: REST API, local photo library APIs (PHPhotoLibrary)
- **Key Features**: Offline photo browsing, background sync, push notifications

#### Android Native App
- **Purpose**: Mobile experience for Android users
- **Technology**: Kotlin, Jetpack Compose, Coroutines
- **Interfaces**: REST API, MediaStore API
- **Key Features**: Background sync, Material Design patterns

#### Progressive Web App (PWA)
- **Purpose**: Desktop/tablet experience, fallback for unsupported devices
- **Technology**: React, TypeScript, TailwindCSS
- **Interfaces**: REST API, File Upload APIs
- **Key Features**: Responsive design, keyboard shortcuts, export functionality

---

### 2. **API Layer**

#### User Service
- **Purpose**: Authentication, user profiles, preferences
- **Technology**: Node.js / Python (FastAPI)
- **Interfaces**: OAuth 2.0, JWT tokens
- **Endpoints**:
  - `POST /auth/login` - User login
  - `GET /users/:id` - Get user profile
  - `PUT /users/:id/preferences` - Update settings

#### Photo Service
- **Purpose**: Photo upload, metadata extraction, storage
- **Technology**: Node.js / Python (FastAPI)
- **Interfaces**: Multipart file upload, cloud storage APIs
- **Endpoints**:
  - `POST /photos/upload` - Upload photos
  - `GET /photos/:id` - Retrieve photo metadata
  - `GET /photos/library` - List all photos for user

#### Circle Service
- **Purpose**: Circle CRUD, sharing, permissions
- **Technology**: Node.js / Python (FastAPI)
- **Interfaces**: REST API, real-time sync (WebSockets)
- **Endpoints**:
  - `POST /circles` - Create circle
  - `POST /circles/:id/invite` - Invite family member
  - `GET /circles/:id/photos` - Get photos for circle

#### Curation Service
- **Purpose**: Trigger curation jobs, retrieve Twelve results
- **Technology**: Python (FastAPI)
- **Interfaces**: REST API, message queue (for async jobs)
- **Endpoints**:
  - `POST /circles/:id/curate` - Trigger curation
  - `GET /circles/:id/twelve/:year` - Get Twelve for year

---

### 3. **AI/ML Layer**

#### Curation Engine
- **Purpose**: Analyze photos, select twelve best per year per circle
- **Technology**: Python, TensorFlow/PyTorch, OpenCV
- **Components**:
  - **Quality Analyzer**: Sharpness, exposure, composition scoring
  - **Emotion Detector**: Facial expression analysis (smiles, hugs)
  - **Diversity Engine**: Temporal/subject spread algorithms
- **Interfaces**: Message queue (receives jobs), blob storage (reads photos)

#### Batch Processor
- **Purpose**: Orchestrate large-scale yearly curation jobs
- **Technology**: Python, Celery (task queue), Redis
- **Interfaces**: Message queue, database (for job status)

---

### 4. **Data Layer**

#### User Database (PostgreSQL)
- **Schema**:
  - `users`: User accounts, auth tokens
  - `circles`: Circle definitions, memberships
  - `circle_members`: Join table for shared circles
  - `preferences`: User settings, notification preferences

#### Photo Metadata Database (PostgreSQL)
- **Schema**:
  - `photos`: Photo metadata (upload date, dimensions, faces)
  - `photo_circles`: Many-to-many mapping (photos → circles)
  - `curation_history`: Record of past curation runs

#### Photo Storage (S3 / Azure Blob)
- **Structure**:
  - Original photos: `s3://photos/{user_id}/{photo_id}.jpg`
  - Thumbnails: `s3://thumbnails/{user_id}/{photo_id}_thumb.jpg`
- **Retention**: Permanent storage with glacier archival for old photos

#### Archive Database (PostgreSQL)
- **Schema**:
  - `twelves`: Yearly Twelve records (circle_id, year, finalized_at)
  - `twelve_photos`: Ordered list of 12 photos per Twelve
  - `exports`: PDF/print export records

---

## Infrastructure

### Hosting

**Cloud Provider**: AWS (primary) or Azure (alternative)

**Services**:
- **Compute**: ECS (Elastic Container Service) for API services
- **Storage**: S3 for photos, RDS (PostgreSQL) for databases
- **Caching**: Redis for session management, API response caching
- **CDN**: CloudFront for fast photo delivery globally
- **ML**: SageMaker for model training/inference (or custom GPU instances)

### Deployment

- **CI/CD**: GitHub Actions for automated testing and deployment
- **Containerization**: Docker for all services
- **Orchestration**: Kubernetes (EKS) for scaling API services
- **Blue-Green Deployments**: Zero-downtime updates

### Monitoring

- **APM**: DataDog or New Relic for application performance monitoring
- **Logging**: Centralized logging (ELK stack or CloudWatch)
- **Alerts**: PagerDuty for critical incidents
- **Analytics**: Mixpanel for user behavior tracking

---

## Security

### Data Protection

- **Encryption at Rest**: AES-256 for all stored photos and metadata
- **Encryption in Transit**: TLS 1.3 for all API calls
- **Photo Privacy**: Photos never used for training AI without explicit consent

### Authentication & Authorization

- **Auth**: OAuth 2.0 + JWT tokens (short-lived, refresh tokens)
- **MFA**: Two-factor authentication available for all users
- **Permissions**: Role-based access control (RBAC) for shared circles

### Compliance

- **GDPR**: Full data export and deletion capabilities
- **CCPA**: User data transparency and opt-out mechanisms
- **Data Residency**: Option to store data in user's region (EU, US, etc.)

---

## Performance

### Targets

- **API Response Time**: <200ms p95 for read operations
- **Photo Upload**: <5 seconds for 10MB photo (on LTE)
- **Timeline Load**: <1 second to load 1 year of Twelves
- **Curation Job**: <30 minutes for 10,000 photos per circle

### Optimization Strategies

- **Lazy Loading**: Load thumbnails first, full-res on demand
- **CDN Caching**: Cache photos close to users globally
- **Database Indexing**: Optimize queries on circle_id, year, user_id
- **Batch Processing**: Curate all circles for a user in one job

---

## Scalability

### Horizontal Scaling

- **API Services**: Stateless; scale with load balancers
- **Curation Jobs**: Queue-based; add workers as needed
- **Database**: Read replicas for query scaling, sharding by user_id if needed

### Cost Optimization

- **Storage Tiering**: Move old photos to glacier storage
- **ML Inference**: Batch processing during off-peak hours
- **CDN**: Aggressive caching to reduce S3 read costs

### Future-Proofing

- **Microservices**: Each component can be replaced independently
- **Open Formats**: Photos stored in JPEG/PNG (never proprietary)
- **Export API**: Users can download all data at any time

---

## Technology Stack Summary

| Layer        | Technology                          |
| ------------ | ----------------------------------- |
| **Frontend** | Swift (iOS), Kotlin (Android), React (Web) |
| **Backend**  | Python (FastAPI), Node.js           |
| **Database** | PostgreSQL, Redis                   |
| **Storage**  | AWS S3, CloudFront CDN              |
| **ML**       | Python, TensorFlow, OpenCV          |
| **Infra**    | Docker, Kubernetes, GitHub Actions  |

---

## Decision Log Reference

See [StackDecisionLog.md](StackDecisionLog.md) for detailed rationale on technology choices.
