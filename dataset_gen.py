"""
Generate realistic training and evaluation datasets for TAESR
Creates data/train.jsonl and data/eval.jsonl with diverse query-document pairs
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

# Set random seed for reproducibility
random.seed(42)


# ============================================================================
# DATASET TEMPLATES - Diverse domains and complexity levels
# ============================================================================

DOMAINS = {
    "science": {
        "easy": [
            ("What is photosynthesis?", 
             "Photosynthesis is the process by which plants convert sunlight into energy. It occurs in chloroplasts and produces glucose and oxygen.",
             "Plants use photosynthesis to create food. This happens in the leaves where chlorophyll absorbs light."),
            
            ("How does gravity work?",
             "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it keeps us grounded and causes objects to fall.",
             "Isaac Newton discovered gravity when an apple fell from a tree. It's the force that pulls things down to Earth."),
            
            ("What are atoms made of?",
             "Atoms are composed of three main particles: protons and neutrons in the nucleus, and electrons orbiting around it.",
             "Everything is made of tiny particles called atoms. They have a center nucleus and electrons spinning around it."),
        ],
        "medium": [
            ("Explain the process of protein synthesis in cells",
             "Protein synthesis involves transcription of DNA to mRNA in the nucleus, followed by translation at ribosomes where tRNA brings amino acids to form polypeptide chains. This process is regulated by various enzymes and occurs in two main stages.",
             "Protein synthesis starts with DNA transcription in the nucleus. The mRNA then travels to ribosomes where amino acids are assembled into proteins through translation."),
            
            ("How do vaccines stimulate immune response?",
             "Vaccines introduce antigens that mimic pathogens, prompting the immune system to produce antibodies and memory cells. This creates adaptive immunity without causing disease, preparing the body for future encounters with the actual pathogen.",
             "When you get vaccinated, your body learns to recognize harmful germs without getting sick. It creates special cells that remember the pathogen."),
            
            ("Describe the carbon cycle in ecosystems",
             "The carbon cycle involves photosynthesis capturing atmospheric CO2, its incorporation into organic compounds, transfer through food chains, and release via respiration and decomposition. Human activities like fossil fuel combustion have disrupted this natural balance.",
             "Carbon moves between atmosphere, living things, and Earth through processes like photosynthesis and respiration. Plants absorb CO2 and animals release it."),
        ],
        "hard": [
            ("Analyze the quantum mechanical model of atomic orbitals and electron configuration",
             "The quantum mechanical model describes electrons in probabilistic wave functions characterized by quantum numbers (n, l, ml, ms). Orbitals represent regions of electron probability density, with shapes determined by angular momentum quantum numbers. The Pauli exclusion principle and Hund's rule govern electron configuration in multi-electron systems.",
             "Quantum mechanics describes where electrons are likely to be found around atoms. Each electron has specific properties defined by quantum numbers and occupies distinct orbital regions."),
            
            ("Explain CRISPR-Cas9 mechanism and its applications in gene editing",
             "CRISPR-Cas9 utilizes guide RNA to direct the Cas9 endonuclease to specific DNA sequences for precise double-strand breaks. The cell's repair mechanisms (NHEJ or HDR) then modify the genome. Applications include disease modeling, therapeutic interventions for genetic disorders, and agricultural improvements through targeted mutagenesis.",
             "CRISPR is a gene editing tool that uses molecular scissors to cut DNA at precise locations. Scientists can then add, remove, or change genetic material."),
            
            ("Discuss thermodynamic principles in biological systems and entropy",
             "Living systems maintain low entropy locally by coupling endergonic reactions to exergonic processes, particularly ATP hydrolysis. The second law of thermodynamics applies, but organisms create order by increasing environmental entropy. Free energy calculations (ΔG = ΔH - TΔS) determine reaction spontaneity in metabolic pathways.",
             "Living things seem to defy entropy by staying organized, but they actually increase disorder in their surroundings by using energy from food."),
        ]
    },
    
    "technology": {
        "easy": [
            ("What is cloud computing?",
             "Cloud computing delivers computing services over the internet, including storage, processing power, and applications. Users can access resources on-demand without managing physical infrastructure.",
             "Cloud computing means storing and accessing data over the internet instead of on your computer's hard drive."),
            
            ("How does WiFi work?",
             "WiFi uses radio waves to transmit data wirelessly between devices and routers. It operates on specific frequency bands (2.4GHz and 5GHz) and follows IEEE 802.11 standards.",
             "WiFi sends information through the air using invisible radio waves, connecting your devices to the internet without cables."),
            
            ("What is machine learning?",
             "Machine learning is a subset of AI where computers learn patterns from data without explicit programming. Algorithms improve performance on tasks through experience.",
             "Machine learning teaches computers to learn from examples, like how humans learn from experience, without being specifically programmed."),
        ],
        "medium": [
            ("Explain how blockchain technology ensures data integrity",
             "Blockchain uses cryptographic hashing and distributed consensus mechanisms to create immutable records. Each block contains a hash of the previous block, timestamp, and transaction data. The decentralized network validates transactions through protocols like proof-of-work or proof-of-stake.",
             "Blockchain creates a secure chain of records that can't be altered because each block is linked to the previous one using cryptography."),
            
            ("Describe the architecture of a RESTful API",
             "RESTful APIs follow architectural constraints including stateless client-server communication, cacheable responses, and uniform interface. They use HTTP methods (GET, POST, PUT, DELETE) to manipulate resources identified by URIs, typically returning JSON or XML data.",
             "REST APIs allow different software systems to communicate over the internet using standard HTTP methods and returning data in formats like JSON."),
            
            ("How does end-to-end encryption protect communications?",
             "End-to-end encryption encrypts messages on the sender's device and decrypts only on the recipient's device using public-key cryptography. Service providers cannot access message content, ensuring privacy even if servers are compromised.",
             "End-to-end encryption scrambles messages so only the sender and receiver can read them. Even the service provider can't see the content."),
        ],
        "hard": [
            ("Analyze the computational complexity of different sorting algorithms and their trade-offs",
             "Quicksort achieves O(n log n) average-case through divide-and-conquer but degrades to O(n²) worst-case. Mergesort guarantees O(n log n) with O(n) space overhead. Heapsort provides O(n log n) in-place sorting. Radix sort achieves O(nk) for integer keys. Algorithm choice depends on data characteristics, memory constraints, and stability requirements.",
             "Different sorting algorithms have different speeds and memory needs. Quicksort is usually fast but can be slow in worst cases, while mergesort is consistently fast but uses more memory."),
            
            ("Explain the Byzantine Generals Problem and its solutions in distributed systems",
             "The Byzantine Generals Problem addresses achieving consensus in distributed systems with faulty or malicious nodes. Solutions include practical Byzantine Fault Tolerance (PBFT) requiring 3f+1 nodes to tolerate f failures, and blockchain consensus mechanisms. These protocols ensure agreement despite arbitrary failures through redundancy and cryptographic verification.",
             "The Byzantine Generals Problem is about getting computers to agree when some might be broken or malicious. Solutions use voting and verification to reach consensus despite failures."),
            
            ("Discuss transformer architecture and attention mechanisms in deep learning",
             "Transformers use self-attention mechanisms to weigh input sequence elements, computing attention scores through query-key-value matrices. Multi-head attention captures different representation subspaces. Positional encoding preserves sequence order. The architecture enables parallel processing and has revolutionized NLP through models like BERT and GPT.",
             "Transformers are neural networks that use attention to focus on important parts of input data. They process entire sequences at once, making them powerful for language tasks."),
        ]
    },
    
    "medicine": {
        "easy": [
            ("What causes the common cold?",
             "The common cold is caused by viral infections, most commonly rhinoviruses. It spreads through respiratory droplets and contact with contaminated surfaces.",
             "Colds are caused by viruses that infect your nose and throat. You catch them by breathing infected air or touching contaminated surfaces."),
            
            ("Why do we need to sleep?",
             "Sleep allows the body to repair tissues, consolidate memories, and regulate hormones. It's essential for physical health, cognitive function, and emotional well-being.",
             "Sleep helps your body rest and repair itself. During sleep, your brain processes information and your body recovers from daily activities."),
            
            ("What is blood pressure?",
             "Blood pressure measures the force of blood against artery walls as the heart pumps. It's expressed as systolic over diastolic pressure, typically around 120/80 mmHg.",
             "Blood pressure is how hard your blood pushes against blood vessel walls. Normal is about 120/80, measured when your heart beats and rests."),
        ],
        "medium": [
            ("Explain how insulin regulates blood glucose levels",
             "Insulin, secreted by pancreatic beta cells, facilitates glucose uptake in muscle and adipose tissue by promoting GLUT4 translocation. It also inhibits hepatic glucose production and stimulates glycogen synthesis. Insulin resistance occurs when cells become less responsive to insulin signaling.",
             "Insulin is a hormone that helps cells absorb sugar from blood. When blood sugar rises, the pancreas releases insulin to bring levels back to normal."),
            
            ("Describe the mechanism of antibiotic resistance in bacteria",
             "Bacteria develop resistance through genetic mutations or horizontal gene transfer. Mechanisms include enzymatic degradation of antibiotics, altered target sites, efflux pumps, and modified cell wall permeability. Overuse and misuse of antibiotics accelerate resistance evolution.",
             "Bacteria become resistant to antibiotics through genetic changes that help them survive drug exposure. This happens faster when antibiotics are overused."),
            
            ("How does the adaptive immune system recognize pathogens?",
             "Adaptive immunity uses T and B lymphocytes with specific antigen receptors generated through V(D)J recombination. T cells recognize peptide-MHC complexes, while B cells produce antibodies against diverse epitopes. Memory cells provide long-lasting immunity after initial exposure.",
             "Your immune system learns to recognize specific germs using special white blood cells. These cells remember pathogens and respond faster if you encounter them again."),
        ],
        "hard": [
            ("Analyze the molecular pathogenesis of Alzheimer's disease and current therapeutic approaches",
             "Alzheimer's pathology involves amyloid-beta plaque accumulation and tau protein hyperphosphorylation leading to neurofibrillary tangles. Neuroinflammation, oxidative stress, and synaptic dysfunction contribute to cognitive decline. Current approaches target amyloid cascade, tau aggregation, and neuroinflammation, though disease-modifying therapies remain limited.",
             "Alzheimer's disease involves abnormal protein buildup in the brain causing nerve cell death. Scientists are developing treatments to clear these proteins and reduce inflammation."),
            
            ("Explain CAR-T cell therapy mechanism and clinical applications in oncology",
             "CAR-T therapy involves genetically engineering patient T cells to express chimeric antigen receptors targeting tumor antigens. These modified cells expand ex vivo before reinfusion, where they recognize and eliminate cancer cells. Approved for B-cell malignancies, challenges include cytokine release syndrome and neurotoxicity management.",
             "CAR-T therapy modifies your own immune cells to fight cancer. Scientists reprogram T cells to recognize and attack cancer cells, showing success in blood cancers."),
            
            ("Discuss epigenetic modifications in gene expression regulation and disease",
             "Epigenetic mechanisms including DNA methylation, histone modifications, and non-coding RNAs regulate gene expression without altering DNA sequence. These modifications are heritable and reversible, influencing development and disease. Aberrant epigenetic patterns contribute to cancer, and epigenetic drugs like HDAC inhibitors show therapeutic promise.",
             "Epigenetics controls which genes are active without changing DNA itself. Chemical tags on DNA and proteins regulate genes, and abnormal patterns can cause diseases like cancer."),
        ]
    },
    
    "business": {
        "easy": [
            ("What is supply and demand?",
             "Supply and demand is an economic model where price is determined by the relationship between product availability and consumer desire. When demand exceeds supply, prices rise; when supply exceeds demand, prices fall.",
             "Supply and demand explains how prices are set. If many people want something but there's not much available, the price goes up."),
            
            ("What is a business plan?",
             "A business plan is a document outlining a company's goals, strategies, target market, financial projections, and operational structure. It serves as a roadmap for business development and is often required for funding.",
             "A business plan describes what your business does, who your customers are, and how you'll make money. It helps guide decisions and attract investors."),
            
            ("What is brand identity?",
             "Brand identity encompasses the visual and verbal elements representing a company, including logo, colors, typography, and messaging. It creates recognition and differentiation in the market.",
             "Brand identity is how a company presents itself to customers through logos, colors, and messaging. It makes the business memorable and distinct."),
        ],
        "medium": [
            ("Explain the concept of sustainable competitive advantage",
             "Sustainable competitive advantage arises from unique resources or capabilities that are valuable, rare, inimitable, and non-substitutable (VRIN framework). Sources include proprietary technology, brand equity, network effects, and organizational culture that competitors cannot easily replicate.",
             "Sustainable competitive advantage means having unique strengths that are hard for competitors to copy, keeping your business ahead long-term."),
            
            ("Describe agile project management methodology",
             "Agile methodology emphasizes iterative development, cross-functional collaboration, and adaptive planning. Teams work in sprints, delivering incremental value with regular stakeholder feedback. Frameworks like Scrum and Kanban facilitate flexibility and continuous improvement.",
             "Agile project management breaks work into short cycles with frequent reviews and adjustments. Teams collaborate closely and adapt quickly to changes."),
            
            ("How does venture capital funding work?",
             "Venture capital involves investors providing capital to high-growth potential startups in exchange for equity. Funding occurs in stages (seed, Series A/B/C) with increasing valuations. VCs seek significant returns through exits via acquisition or IPO.",
             "Venture capitalists invest money in startups with high growth potential in exchange for ownership shares. They provide funding in stages as the company grows."),
        ],
        "hard": [
            ("Analyze Porter's Five Forces framework and its application in competitive strategy",
             "Porter's Five Forces analyzes industry attractiveness through threat of new entrants, bargaining power of suppliers/buyers, threat of substitutes, and competitive rivalry. Each force influences profitability and strategic positioning. Firms can build competitive advantage by understanding and mitigating these forces through differentiation, cost leadership, or focus strategies.",
             "Porter's Five Forces helps businesses understand competition by examining five factors that affect profitability: new competitors, supplier power, buyer power, substitutes, and rivalry."),
            
            ("Explain the efficient market hypothesis and its implications for investment strategy",
             "The efficient market hypothesis posits that asset prices reflect all available information, making it impossible to consistently achieve above-market returns. The semi-strong form suggests technical and fundamental analysis cannot beat the market. Implications include passive indexing strategies and challenges to active management fees.",
             "The efficient market hypothesis says stock prices already include all known information, making it very hard to consistently beat the market through stock picking."),
            
            ("Discuss the principal-agent problem in corporate governance",
             "The principal-agent problem arises when managers (agents) may not act in shareholders' (principals') best interests due to misaligned incentives and information asymmetry. Solutions include performance-based compensation, board oversight, and market disciplines. Agency costs reduce firm value through monitoring expenses and residual loss.",
             "The principal-agent problem occurs when company managers might not act in shareholders' best interests. Companies use incentives and oversight to align their goals."),
        ]
    },
    
    "history": {
        "easy": [
            ("When did World War II end?",
             "World War II ended in 1945, with Germany surrendering in May and Japan surrendering in September after atomic bombs were dropped on Hiroshima and Nagasaki.",
             "World War II ended in 1945. Germany surrendered in May, and Japan surrendered in September after the atomic bombings."),
            
            ("Who was the first president of the United States?",
             "George Washington was the first president of the United States, serving from 1789 to 1797. He is known as the 'Father of His Country.'",
             "George Washington became the first U.S. president in 1789 and served two terms. He helped establish many presidential traditions."),
            
            ("What was the Renaissance?",
             "The Renaissance was a cultural movement in Europe from the 14th to 17th century, marking the transition from medieval to modern times. It featured renewed interest in classical learning, art, and humanism.",
             "The Renaissance was a period of cultural rebirth in Europe when art, science, and learning flourished. It began in Italy in the 1300s."),
        ],
        "medium": [
            ("Explain the causes of the French Revolution",
             "The French Revolution (1789-1799) stemmed from economic crisis, social inequality under the ancien régime, Enlightenment ideas challenging absolute monarchy, and fiscal mismanagement. The bourgeoisie sought political representation while peasants suffered from feudal obligations and food shortages.",
             "The French Revolution was caused by financial crisis, unfair social classes, new ideas about rights and equality, and widespread hunger among common people."),
            
            ("Describe the impact of the Industrial Revolution on society",
             "The Industrial Revolution transformed agrarian societies into industrial ones through mechanization and factory systems. It caused urbanization, created industrial working class, improved productivity but initially worsened working conditions, and led to significant social and economic changes including rise of capitalism.",
             "The Industrial Revolution changed how people worked and lived. Factories replaced farms, cities grew rapidly, and new technologies improved production but created challenging working conditions."),
            
            ("How did the Silk Road facilitate cultural exchange?",
             "The Silk Road connected East and West, enabling trade of silk, spices, and ideas across Asia. It facilitated spread of Buddhism, Islam, and Christianity, exchange of technologies like papermaking and gunpowder, and cross-cultural artistic influences.",
             "The Silk Road was a network of trade routes connecting Asia with Europe and Africa. Merchants exchanged goods, ideas, religions, and technologies along these routes."),
        ],
        "hard": [
            ("Analyze the geopolitical consequences of the Treaty of Versailles",
             "The Treaty of Versailles imposed harsh reparations on Germany, territorial losses, and war guilt clause, creating economic hardship and resentment. It redrew European borders without considering ethnic composition, established League of Nations, and inadvertently set conditions for WWII by destabilizing Weimar Republic and enabling Hitler's rise.",
             "The Treaty of Versailles ended WWI but punished Germany severely. This created economic problems and resentment that contributed to Hitler's rise and WWII."),
            
            ("Discuss the historiographical debates surrounding the fall of the Roman Empire",
             "Historians debate whether Rome 'fell' or transformed. Theories include barbarian invasions, economic decline, overextension, Christian influence weakening civic virtue, lead poisoning, and climate change. Modern scholarship emphasizes gradual transformation rather than sudden collapse, with regional variations in Western and Eastern empires.",
             "Historians disagree about why the Roman Empire ended. Some blame invasions, others economic problems or overexpansion. Modern scholars see gradual change rather than sudden collapse."),
            
            ("Examine the role of imperialism in shaping the modern Middle East",
             "Post-WWI Sykes-Picot Agreement and British-French mandates created artificial borders ignoring ethnic and religious divisions. Balfour Declaration supported Jewish homeland, creating Israeli-Palestinian conflict. Oil discoveries intensified Western involvement. Legacy includes authoritarianism, sectarian conflicts, and ongoing regional instability.",
             "After WWI, European powers divided the Middle East with artificial borders that ignored local groups. This, along with oil interests and the creation of Israel, shaped modern conflicts."),
        ]
    }
}


def generate_dataset(num_samples: int, include_negatives: bool = True) -> List[Dict]:
    """
    Generate synthetic dataset with realistic query-document pairs.
    
    Args:
        num_samples: Number of examples to generate
        include_negatives: Whether to include hard negative examples
    
    Returns:
        List of dataset examples
    """
    dataset = []
    domains = list(DOMAINS.keys())
    complexities = ["easy", "medium", "hard"]
    
    for _ in range(num_samples):
        # Select random domain and complexity
        domain = random.choice(domains)
        complexity = random.choice(complexities)
        
        # Get templates for this domain/complexity
        templates = DOMAINS[domain][complexity]
        query, positive, negative_pool = random.choice(templates)
        
        example = {
            "query": query,
            "positive": positive,
            "complexity": complexity,
            "domain": domain
        }
        
        # Add negative example from different context
        if include_negatives:
            # Get negative from different domain or complexity
            neg_domain = random.choice([d for d in domains if d != domain])
            neg_complexity = random.choice(complexities)
            neg_templates = DOMAINS[neg_domain][neg_complexity]
            _, neg_doc, _ = random.choice(neg_templates)
            example["negative"] = neg_doc
        
        dataset.append(example)
    
    return dataset


def augment_with_variations(dataset: List[Dict]) -> List[Dict]:
    """
    Augment dataset with paraphrases and variations.
    """
    augmented = dataset.copy()
    
    # Add paraphrased versions
    paraphrase_prefixes = [
        "Can you explain ",
        "I need to understand ",
        "Help me with ",
        "Looking for information about ",
        "Tell me about ",
    ]
    
    for example in dataset[:len(dataset)//3]:  # Augment 1/3 of data
        if random.random() > 0.5:
            prefix = random.choice(paraphrase_prefixes)
            augmented.append({
                "query": prefix + example["query"].lower(),
                "positive": example["positive"],
                "complexity": example["complexity"],
                "domain": example["domain"]
            })
    
    return augmented


def save_dataset(dataset: List[Dict], filepath: str):
    """Save dataset in JSONL format."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(dataset)} examples to {filepath}")


def analyze_dataset(dataset: List[Dict], name: str):
    """Print dataset statistics."""
    print(f"\n📊 Dataset Analysis: {name}")
    print(f"   Total examples: {len(dataset)}")
    
    # Complexity distribution
    complexity_counts = {}
    for ex in dataset:
        comp = ex.get('complexity', 'unknown')
        complexity_counts[comp] = complexity_counts.get(comp, 0) + 1
    
    print(f"   Complexity distribution:")
    for comp, count in sorted(complexity_counts.items()):
        print(f"      {comp}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Domain distribution
    domain_counts = {}
    for ex in dataset:
        dom = ex.get('domain', 'unknown')
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
    
    print(f"   Domain distribution:")
    for dom, count in sorted(domain_counts.items()):
        print(f"      {dom}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Average lengths
    avg_query_len = sum(len(ex['query'].split()) for ex in dataset) / len(dataset)
    avg_pos_len = sum(len(ex['positive'].split()) for ex in dataset) / len(dataset)
    
    print(f"   Average query length: {avg_query_len:.1f} words")
    print(f"   Average positive length: {avg_pos_len:.1f} words")


def main():
    """Generate training and evaluation datasets."""
    print("="*80)
    print("TAESR Dataset Generator")
    print("="*80)
    
    # Generate training set (larger, with augmentation)
    print("\n🔨 Generating training dataset...")
    train_dataset = generate_dataset(num_samples=5000, include_negatives=True)
    train_dataset = augment_with_variations(train_dataset)
    random.shuffle(train_dataset)
    save_dataset(train_dataset, "data/train.jsonl")
    analyze_dataset(train_dataset, "Training Set")
    
    # Generate evaluation set (smaller, no augmentation)
    print("\n🔨 Generating evaluation dataset...")
    eval_dataset = generate_dataset(num_samples=500, include_negatives=True)
    save_dataset(eval_dataset, "data/eval.jsonl")
    analyze_dataset(eval_dataset, "Evaluation Set")
    
    # Generate a small test set for quick testing
    print("\n🔨 Generating test dataset...")
    test_dataset = generate_dataset(num_samples=100, include_negatives=False)
    save_dataset(test_dataset, "data/test.jsonl")
    analyze_dataset(test_dataset, "Test Set")
    
    print("\n" + "="*80)
    print("✅ Dataset generation complete!")
    print("="*80)
    print("\nGenerated files:")
    print("   📁 data/train.jsonl - Training set with augmentation")
    print("   📁 data/eval.jsonl  - Evaluation set for validation")
    print("   📁 data/test.jsonl  - Small test set for quick checks")
    print("\nExample record format:")
    print(json.dumps(train_dataset[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()