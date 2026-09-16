# ALFWorld Qwen3-8B GiGPO — optimizer step 1 episode reconstruction

> Important: `step` in the raw JSONL is the **optimizer step**. Each JSONL row is one environment-turn training sample, not one whole ALFWorld episode. Episodes below are reconstructed from the environment-step counter inside `input`.

## Reconstructed batch

- raw rows: **5120**
- reconstructed episodes: **1685**
- unique tasks: **16**
- task group-size distribution: `{24: 1, 25: 1, 39: 1, 47: 1, 53: 1, 58: 1, 62: 1, 67: 1, 101: 1, 126: 1, 141: 1, 146: 1, 187: 1, 191: 1, 203: 1, 215: 1}`
- episode outcomes: `{'success': 297, 'failure': 1388}`
- episode score distribution: `{'10.0': 220, '9.899999618530273': 77, '0.0': 794, '-0.10000000149011612': 594}`
- action length min/median/mean/max: **0 / 3 / 6.71 / 49**
- parsed actions: **4258 / 5120**
- non-admissible emitted actions (where check possible): **89**
- tasks containing both success and failure episodes: **7**

## Highest-leverage success/failure pairs

For each task, this selects the success/failure pair sharing the longest identical action prefix. These are useful candidate intervention points because the trajectories stay behaviorally matched as long as possible before diverging.

### 1. cool some lettuce and put it in garbagecan.

- task group size: **47**
- success episode / score: **1251 / 10.0**
- failure episode / score: **1646 / 0.0**
- shared action prefix: **11 step(s)**
- first divergence: **step 12**
- success action: `None`
- failure action: `go to cabinet 1`
- failed action admissible: **None**
- failed action repeats a recent action: **False**

**Observation on success branch at divergence**
```text
None
```
**Observation on failure branch at divergence**
```text
You arrive at countertop 1. On the countertop 1, you see a bowl 3, a bread 1, a cup 1, a dishsponge 1, a houseplant 2, a houseplant 1, a knife 1, a ladle 2, a pan 2, and a papertowelroll 1.
```

### 2. heat some apple and put it in garbagecan.

- task group size: **101**
- success episode / score: **0 / 10.0**
- failure episode / score: **18 / 0.0**
- shared action prefix: **10 step(s)**
- first divergence: **step 16**
- success action: `ernatively, maybe the apple is`
- failure action: `go to cabinet 1`
- failed action admissible: **None**
- failed action repeats a recent action: **False**

**Observation on success branch at divergence**
```text
You arrive at countertop 1. On the countertop 1, you see a butterknife 1, a dishsponge 1, a houseplant 1, and a peppershaker 1.
```
**Observation on failure branch at divergence**
```text
You arrive at countertop 1. On the countertop 1, you see a butterknife 1, a dishsponge 1, a houseplant 1, and a peppershaker 1.
```

### 3. clean some potato and put it in garbagecan.

- task group size: **39**
- success episode / score: **128 / 10.0**
- failure episode / score: **508 / 0.0**
- shared action prefix: **8 step(s)**
- first divergence: **step 27**
- success action: `go to sinkbasin 1`
- failure action: `ink> <action>go to sinkbasin`
- failed action admissible: **None**
- failed action repeats a recent action: **False**

**Observation on success branch at divergence**
```text
You arrive at fridge 1. The fridge 1 is open. In it, you see a bowl 2, a bowl 1, a egg 1, a mug 2, and a tomato 1.
```
**Observation on failure branch at divergence**
```text
You arrive at fridge 1. The fridge 1 is open. In it, you see a bowl 2, a bowl 1, a egg 1, a mug 2, and a tomato 1.
```

### 4. put a book in desk.

- task group size: **58**
- success episode / score: **1179 / 10.0**
- failure episode / score: **773 / 0.0**
- shared action prefix: **4 step(s)**
- first divergence: **step 25**
- success action: `go to drawer 3`
- failure action: `go to shelf 2`
- failed action admissible: **None**
- failed action repeats a recent action: **False**

**Observation on success branch at divergence**
```text
You arrive at drawer 2. The drawer 2 is open. In it, you see nothing.
```
**Observation on failure branch at divergence**
```text
You arrive at shelf 1. On the shelf 1, you see a alarmclock 1, a box 1, a creditcard 2, a houseplant 1, and a pen 1.
```

### 5. put a clean dishsponge in countertop.

- task group size: **203**
- success episode / score: **346 / 10.0**
- failure episode / score: **357 / 0.0**
- shared action prefix: **3 step(s)**
- first divergence: **step 1**
- success action: `None`
- failure action: `go to sinkbasin 1`
- failed action admissible: **None**
- failed action repeats a recent action: **False**

**Observation on success branch at divergence**
```text
None
```
**Observation on failure branch at divergence**
```text
-= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a shelf 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, a towelholder 2, and a towelholder 1. Your task is to: put a clean dishsponge in countertop.
```

### 6. put a cool bowl in shelf.

- task group size: **62**
- success episode / score: **539 / 9.899999618530273**
- failure episode / score: **534 / 0.0**
- shared action prefix: **2 step(s)**
- first divergence: **step 1**
- success action: `None`
- failure action: `go to fridge 1`
- failed action admissible: **None**
- failed action repeats a recent action: **False**

**Observation on success branch at divergence**
```text
None
```
**Observation on failure branch at divergence**
```text
-= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1. Your task is to: put a cool bowl in shelf.
```

### 7. put a watch in coffeetable.

- task group size: **53**
- success episode / score: **310 / 10.0**
- failure episode / score: **741 / 0.0**
- shared action prefix: **2 step(s)**
- first divergence: **step 3**
- success action: `go to dresser 1`
- failure action: `go to sidetable 2`
- failed action admissible: **True**
- failed action repeats a recent action: **False**

**Observation on success branch at divergence**
```text
You arrive at sidetable 1. On the sidetable 1, you see nothing.
```
**Observation on failure branch at divergence**
```text
You arrive at sidetable 1. On the sidetable 1, you see nothing.
```
