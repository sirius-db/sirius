# Task FSM

Source: [src/task.rs](src/task.rs)

```mermaid
flowchart LR
    start((entry))
    created["Created"]
    queued["Queued<br/>uses: TaskQueue"]
    routing["Routing<br/>uses: TaskManagerLoopThread"]
    reserving["Reserving<br/>uses: TaskManagerLoopThread"]
    downgrading["Downgrading<br/>uses: TaskManagerLoopThread"]
    preparing["Preparing<br/>uses: ExecutorThread"]
    computing["Computing<br/>uses: ExecutorThread"]
    finalizing["Finalizing"]
    exit((exit))

    start --> created
    created -->|enters first scheduling queue| queued
    queued -->|manager pops queued task| routing
    routing -->|scan/source reserves in manager loop| reserving
    reserving -->|reservation succeeded| preparing
    preparing -->|worker preparation complete| computing
    computing -->|normal completion or execution failure| finalizing
    finalizing --> exit

    routing -->|GPU task routed to executor queue| queued
    queued -->|GPU task popped from executor queue| reserving
    reserving -->|reservation shortfall| downgrading
    downgrading -->|retry after downgrade| reserving
    computing -->|next operator event| computing

    created -.->|created but never queued| finalizing
    queued -.->|drain, interrupted scheduling, or cancellation| finalizing
    routing -.->|executor enqueue failed or dropped| finalizing
    reserving -.->|reservation failed| finalizing
    downgrading -.->|downgrade cancelled or failed| finalizing
    preparing -.->|tier upgrade failed| finalizing
```

Solid transitions are the regular scheduling/execution path. Dashed transitions
are cleanup, cancellation, or failure paths that terminate the FSM through
`finalizing`.
