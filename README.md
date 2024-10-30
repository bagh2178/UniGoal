# UniGoal

## Installation of UniGoal

**Step 1:** Create new conda environment ``conda create -n unigoal``

**Step 2:** Install the requirements.txt ``pip install -r requirements.txt``

**Step 3:** Install IEVE according to [here](https://github.com/XiaohanLei/IEVE)

**Step 4:** Install ConceptGraph according to [here](https://github.com/concept-graphs/concept-graphs)

**Step 5:** Install Grounded-SAM according to [here](https://github.com/IDEA-Research/Grounded-Segment-Anything)

**Step 6:** Change the ``sys.path.append(xxx)`` in ``./main.py`` and ``./scenegraph/scenegraph.py`` to your workspace path

## Installation of ModelServer

**Step 1:** Install ModelServer in a new conda environment according to [here](https://github.com/bagh2178/ModelServer)

## Evaluation of UniGoal

**Step 1:** Start ModelServer according to [here](https://github.com/bagh2178/ModelServer)

**Step 2:** Run UniGoal on a single GPU ``CUDA_VISIBLE_DEVICES=0 python main.py``
