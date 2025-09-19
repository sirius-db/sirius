# Glossary

This document contains definitions of key terms and concepts used throughout the Sirius project.

## Navigation

[A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [J](#j) | [K](#k) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [Q](#q) | [R](#r) | [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w) | [X](#x) | [Y](#y) | [Z](#z)

## A

## B

## C

## D

**Data Repository** - A container for Data Batches where all Scan Executors and Pipeline Executor tasks get output to. Data in the repository can have its underlying source of data moved between memory tiers.

**Downgrade Executor** - An abstraction that has a thread pool for copying data from GPU to CPU and a queue of downgrade tasks.

**Downgrade Task** - A task that consists of a Data Batch whose memory resides in GPU that we are going to move to CPU.

## E

## F

## G

**GPU Thread** - A thread which has a stream associated with it and can be used to execute tasks on a GPU. It pulls from the pipeline queue in order to get tasks to process.

**GPU Thread Pool** - A pool of GPU threads that exist in the pipeline executor and are used to process tasks as they are added to the queue. The number of threads here define the parallelism of gpu compute.

## H

## I

## J

## K

**Kernel** - A CUDA compute kernel.

## L

## M

**Memory Reservation** - A lease on memory. It means you are expected to be consuming that amount of memory during your execution.

**Memory Reservation Manager** - A system that can give memory reservations into one or more memory tiers. Its purpose is to be able to block executing threads from proceeding when the reservations are full utilized.

## N

## O

**Operators** - Parts of the physical plan that are used to build a pipeline. Operators that can be pipelined are used to create a pipeline that then gets wrapped into a task and executed.

## P

**Pipeline** - A way of describing the transformations that a task is going to be applying to input data. It is built up by chaining the execute method of multiple operators together.

**Pipeline Executor** - An executor that uses GPUs for execution in its backend.

**Pipeline Task Queue** - A task queue that stores tasks that are going to be processed on the GPU by Sirius using the Pipeline executor. All Pipeline executor tasks are added to this queue for execution by the operators.

## Q

## R

## S

**Scan Executor** - An executor that uses DuckDB to scan data from sources and push into the Data Repository.

**Scan Task Queue** - A Task Queue which uses the regular DuckDB execution model for storing tasks related to scanning data into the system.

## T

## U

## V

## W

## X

## Y

## Z

---

