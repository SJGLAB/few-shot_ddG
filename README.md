# An integrated few-shot learning framework for prediction of mutation-induced protein stability changes using complementary computational information

This is **a combinatorial framework of four representative methods:** ESM2-650M, FoldX, ProteinMPNN-ddG, and Rapid Protein Stability Prediction (RASP). Different computational methods for predicting mutation-induced changes in protein stability capture distinct aspects of protein properties. Integrating these complementary sources of information may therefore improve prediction accuracy.

## ✨ Datasets

* **MegaScale**:  a large-scale dataset of protein mutations with experimentally measured thermal stability containing training, testing and validation sets. The training set contains 215,731 mutations across 239 proteins, while the validation and test sets contain 27,328 mutations across 31 proteins and 28,172 mutations across 28 proteins, respectively.

* **S571**: 571 single-point mutations across 37 proteins.

* **S669**: 669 single-point mutations across 94 proteins.

* **S2648**: 2648 single-point mutations across 131 proteins.

## 📝 Individual methods

* **ESM2**: a protein language model based on the transformer architecture, pretrained on large-scale protein sequence datasets to learn representations that capture evolutionary patterns.

* **FoldX**: a physics-based protein stability prediction method that estimates the energetic effects of amino acid substitutions using an empirical force field. Its energy function incorporates multiple contributions, including electrostatic interactions, hydrogen bonding, van der Waals interactions, and entropic effects.

* **ProteinMPNN**: a deep learning model originally developed for the protein inverse folding problem. It employs a message-passing neural network architecture, representing amino acid residues as graph nodes and their spatial relationships as edges, to capture structural and inter-residue interaction information. **ProteinMPNN-ddG** extends this framework to protein stability prediction by incorporating a correction term associated with the background occurrence probabilities of amino acids.

* **RASP**: a convolutional neural network using Rosetta-derived energy-based estimates as training targets, enabling rapid stability prediction without the need for computationally expensive energy calculations for each mutation.

## 🛠️ Training and Validation

The training and validation processes can be easily replicated using the provided scripts.

```bash
cd datasets/megascale/four/

### Training process
python four.py

### Validation process
python calculate_val.py
python calculate_test.py
```

## 📊 Figures and Tables

All the figures and tables mentioned in the articles can be found in the corresponding sections.

## Contact

For any questions, bug reports, or feature requests, please feel free to reach out:

* **Email:** [mcong_wu@163.com](mailto:mcong_wu@163.com)

