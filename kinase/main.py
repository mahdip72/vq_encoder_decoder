from transformers import EsmTokenizer, EsmModel
import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

# Load the tokenizer and model
model_name = "facebook/esm2_t12_35M_UR50D"  # Example model, change as needed
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name).cuda()


def get_protein_embedding(sequence):
    # Tokenize the sequence
    inputs = tokenizer(sequence, return_tensors="pt")
    # move inputs to the GPU
    inputs = {key: value.cuda() for key, value in inputs.items()}
    # Get the embeddings
    with torch.inference_mode():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output.cpu()

    return embeddings


def get_many_embeddings(prot_dict, max_length=2048, progress_bar=True):
    """
    Extract the embeddings of a dictionary of protein sequences.
    :param prot_dict: dictionary of protein sequences (name: sequence)
    :param max_length: maximum sequence length to consider
    :param progress_bar: if True, display a progress bar
    :return embedding_tensor: [N_samples, embedding_size] tensor of protein embeddings (name: embedding)
    """
    embedding_list = []
    progress_bar = tqdm(prot_dict, disable=not progress_bar)
    progress_bar.set_description("Extracting embeddings")

    for name in progress_bar:
        sequence = prot_dict[name]
        # Trim sequences that are longer than 2048 residues
        sequence = sequence[:2048]
        prot_embedding = get_protein_embedding(sequence)
        embedding_list.append(prot_embedding.squeeze(0))

    embedding_tensor = torch.stack(embedding_list)
    return embedding_tensor


def get_unique_kinases(kinase_df):
    """
    Get the unique kinase sequences from a dataset.
    :param kinase_df: DataFrame of kinase data
    :return kinase_dict: dictionary of unique name:sequence kinase pairs
    """
    # Drop duplicates to get unique kinase samples
    unique_df = kinase_df.drop_duplicates(subset='kinase')

    # Create a dictionary with kinase as keys and kinase_sequence as values
    kinase_dict = unique_df.set_index('kinase')['kinase_sequence'].to_dict()

    return kinase_dict


def calc_embedding_distance_map(embeddings, distance_type='euclidean'):
    """
    Calculate the distance map for a tensor of embeddings.
    :param embeddings: [N_samples, embedding_size] tensor of embeddings
    :param distance_type: 'euclidean' or 'cosine'
    :return distance_map: [N_samples, N_samples] distance map of embeddings
    """

    if distance_type == 'euclidean':
        # Expand dimensions to compute pairwise differences
        expanded_embeddings = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        # Calculate Euclidean distances
        distance_map = torch.norm(expanded_embeddings, p=2, dim=2)

    elif distance_type == 'cosine':
        # Create a CosineSimilarity module
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-8)
        # Expand dimensions to compute pairwise cosine similarity
        embeddings1 = embeddings.unsqueeze(1).expand(-1, embeddings.size(0), -1)
        embeddings2 = embeddings.unsqueeze(0).expand(embeddings.size(0), -1, -1)
        # Calculate cosine similarities
        cosine_similarity = cos(embeddings1, embeddings2)
        # Convert cosine similarity to cosine distance
        distance_map = 1 - cosine_similarity

    else:
        raise ValueError('Distance type must be either "euclidean" or "cosine"')

    return distance_map


def get_k_nearest_neighbors(distance_map, k=1):
    """
    Get the k-nearest-neighbors for each sample of a distance map.
    :param distance_map: [N_samples, N_samples] distance map
    :param k: number of neighbors to get
    :return k_nearest: [N_samples, k] indices of nearest neighbors for each sample
    """
    # Ensure the distance map is a square matrix
    assert distance_map.size(0) == distance_map.size(1), "distance_map must be a square matrix"

    # Get the indices of the k smallest distances for each sample
    k_nearest = torch.argsort(distance_map, dim=1)[:, 1:k + 1]

    return k_nearest


def get_negative_kinase_distance_pairs(kinase_df, k=1, max_length=2048, distance_type='euclidean', progress_bar=True):
    """
    Get k negative kinase distance pairs from a dataframe.
    :param kinase_df: DataFrame of kinase data
    :param k: number of pairs to get for each kinase sample
    :param max_length: maximum sequence length to consider
    :param distance_type: 'euclidean' or 'cosine'
    :param progress_bar: if True, display a progress bar
    :return negative_pairs: [N_samples, k]
    """
    kinase_dict = get_unique_kinases(kinase_df)
    embeddings = get_many_embeddings(kinase_dict, progress_bar=progress_bar)
    distance_map = calc_embedding_distance_map(embeddings, distance_type=distance_type)
    k_nearest = get_k_nearest_neighbors(distance_map, k=k)
    negative_pairs = torch.gather(distance_map, 1, k_nearest)
    return negative_pairs


def get_negative_kinase_name_pairs(kinase_df, k=1, max_length=2048, distance_type='euclidean', progress_bar=True):
    """
    Get k negative kinase name pairs from a dataframe.
    :param kinase_df: DataFrame of kinase data
    :param k: number of pairs to get for each kinase sample
    :param max_length: maximum sequence length to consider
    :param distance_type: 'euclidean' or 'cosine'
    :param progress_bar: if True, display a progress bar
    :return kinase_name_pairs: dictionary of kinase name pairs (name:list[k nearest names])
    """
    kinase_dict = get_unique_kinases(kinase_df)
    embeddings = get_many_embeddings(kinase_dict, progress_bar=progress_bar)
    distance_map = calc_embedding_distance_map(embeddings, distance_type=distance_type)
    k_nearest = get_k_nearest_neighbors(distance_map, k=k)

    # Get the names corresponding to the k nearest neighbors
    kinase_name_list = list(kinase_dict.values())
    kinase_name_pairs = deepcopy(kinase_dict)
    for i, name in enumerate(kinase_name_pairs):
        neighbor_names = []
        for neighbor_idx in k_nearest[i]:
            neighbor_names.append(kinase_name_list[neighbor_idx])
        kinase_name_pairs[name] = neighbor_names

    return kinase_name_pairs


def main(pandas_df):
    return None


if __name__ == '__main__':
    # Example usage
    protein_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAYVLSMSPARGCVTRDCRVCTRVYADRTKFGINPQTFRYYTDRVRFDG"  # Replace with your sequence
    embedding = get_protein_embedding(protein_sequence)
    print(embedding.shape)

    data_path = "/DATA/renjz/data/train_filtered.csv"
    df = pd.read_csv(data_path)

    kinase_seq_dict = get_unique_kinases(df)
    # kinase_seq_dict = {"A": "AAA",
    #                    "B": "BAA",
    #                    "F": "AAAAAAAAAAAAAAAAAAAAAA",
    #                    "G": "AAAAAAAAAAAAAAAAAAAAAB",
    #                    "H": "GOIJRWORIRGJIPJPGPRCMI",
    #                    "ChaK1": "MSQKSWIESTLTKRECVYIIPSSKDPHRCLPGCQICQQLVRCFCGRLVKQHACFTASLAMKYSDVKLGDHFNQAIEEWSVEKHTEQSPTDAYGVINFQGGSHSYRAKYVRLSYDTKPEVILQLLLKEWQMELPKLVISVHGGMQKFELHPRIKQLLGKGLIKAAVTTGAWILTGGVNTGVAKHVGDALKEHASRSSRKICTIGIAPWGVIENRNDLVGRDVVAPYQTLLNPLSKLNVLNNLHSHFILVDDGTVGKYGAEVRLRRELEKTINQQRIHARIGQGVPVVALIFEGGPNVILTVLEYLQESPPVPVVVCEGTGRAADLLAYIHKQTEEGGNLPDAAEPDIISTIKKTFNFGQNEALHLFQTLMECMKRKELITVFHIGSDEHQDIDVAILTALLKGTNASAFDQLILTLAWDRVDIAKNHVFVYGQQWLVGSLEQAMLDALVMDRVAFVKLLIENGVSMHKFLTIPRLEELYNTKQGPTNPMLFHLVRDVKQGNLPPGYKITLIDIGLVIEYLMGGTYRCTYTRKRFRLIYNSLGGNNRRSGRNTSSSTPQLRKSHESFGNRADKKEKMRHNHFIKTAQPYRPKIDTVMEEGKKKRTKDEIVDIDDPETKRFPYPLNELLIWACLMKRQVMARFLWQHGEESMAKALVACKIYRSMAYEAKQSDLVDDTSEELKQYSNDFGQLAVELLEQSFRQDETMAMKLLTYELKNWSNSTCLKLAVSSRLRPFVAHTCTQMLLSDMWMGRLNMRKNSWYKVILSILVPPAILLLEYKTKAEMSHIPQSQDAHQMTMDDSENNFQNITEEIPMEVFKEVRILDSNEGKNEMEIQMKSKKLPITRKFYAFYHAPIVKFWFNTLAYLGFLMLYTFVVLVQMEQLPSVQEWIVIAYIFTYAIEKVREIFMSEAGKVNQKIKVWFSDYFNISDTIAIISFFIGFGLRFGAKWNFANAYDNHVFVAGRLIYCLNIIFWYVRLLDFLAVNQQAGPYVMMIGKMVANMFYIVVIMALVLLSFGVPRKAILYPHEAPSWTLAKDIVFHPYWMIFGEVYAYEIDVCANDSVIPQICGPGTWLTPFLQAVYLFVQYIIMVNLLIAFFNNVYLQVKAISNIVWKYQRYHFIMAYHEKPVLPPPLIILSHIVSLFCCICKRRKKDKTSDGPKLFLTEEDQKKLHDFEEQCVEMYFNEKDDKFHSGSEERIRVTFERVEQMCIQIKEVGDRVNYIKRSLQSLDSQIGHLQDLSALTVDTLKTLTAQKASEASKVHNEITRELSISKHLAQNLIDDGPVRPSVWKKHGVVNTLSSSLPQGDLESNNPFHCNILMKDDKDPQCNIFGQDLPAVPQRKEFNFPEAGSSSGALFPSAVSPPELRQRLHGVELLKIFNKNQKLGSSSTSIPHLSSPPTKFFVSTPSQPSCKSHLETGTKDQETVCSKATEGDNTEFGAFVGHRDSMDLQRFKETSNKIKILSNNNTSENTLKRVSSLAGFTDCHRTSIPVHSKQAEKISRRPSTEDTHEVDSKAALIPDWLQDRPSNREMPSEEGTLNGLTSPFKPAMDTNYYYSAVERNNLMRLSQSIPFTPVPPRGEPVTVYRLEESSPNILNNSMSSWSQLGLCAKIEFLSKEEMGGGLRRAVKVQCTWSEHDILKSGHLYIIKSFLPEVVNTWSSIYKEDTVLHLCLREIQQQRAAQKLTFAFNQMKPKSIPYSPRFLEVFLLYCHSAGQWFAVEECMTGEFRKYNNNNGDEIIPTNTLEEIMLAFSHWTYEYTRGELLVLDLQGVGENLTDPSVIKAEEKRSCDMVFGPANLGEDAIKNFRAKHHCNSCCRKLKLPDLKRNDYTPDKIIFPQDEPSDLNLQPGNSTKESESTNSVRLML",
    #                    "ATM": "MSLVLNDLLICCRQLEHDRATERKKEVEKFKRLIRDPETIKHLDRHSDSKQGKYLNWDAVFRFLQKYIQKETECLRIAKPNVSASTQASRQKKMQEISSLVKYFIKCANRRAPRLKCQELLNYIMDTVKDSSNGAIYGADCSNILLKDILSVRKYWCEISQQQWLELFSVYFRLYLKPSQDVHRVLVARIIHAVTKGCCSQTDGLNSKFLDFFSKAIQCARQEKSSSGLNHILAALTIFLKTLAVNFRIRVCELGDEILPTLLYIWTQHRLNDSLKEVIIELFQLQIYIHHPKGAKTQEKGAYESTKWRSILYNLYDLLVNEISHIGSRGKYSSGFRNIAVKENLIELMADICHQVFNEDTRSLEISQSYTTTQRESSDYSVPCKRKKIELGWEVIKDHLQKSQNDFDLVPWLQIATQLISKYPASLPNCELSPLLMILSQLLPQQRHGERTPYVLRCLTEVALCQDKRSNLESSQKSDLLKLWNKIWCITFRGISSEQIQAENFGLLGAIIQGSLVEVDREFWKLFTGSACRPSCPAVCCLTLALTTSIVPGTVKMGIEQNMCEVNRSFSLKESIMKWLLFYQLEGDLENSTEVPPILHSNFPHLVLEKILVSLTMKNCKAAMNFFQSVPECEHHQKDKEELSFSEVEELFLQTTFDKMDFLTIVRECGIEKHQSSIGFSVHQNLKESLDRCLLGLSEQLLNNYSSEITNSETLVRCSRLLVGVLGCYCYMGVIAEEEAYKSELFQKAKSLMQCAGESITLFKNKTNEEFRIGSLRNMMQLCTRCLSNCTKKSPNKIASGFFLRLLTSKLMNDIADICKSLASFIKKPFDRGEVESMEDDTNGNLMEVEDQSSMNLFNDYPDSSVSDANEPGESQSTIGAINPLAEEYLSKQDLLFLDMLKFLCLCVTTAQTNTVSFRAADIRRKLLMLIDSSTLEPTKSLHLHMYLMLLKELPGEEYPLPMEDVLELLKPLSNVCSLYRRDQDVCKTILNHVLHVVKNLGQSNMDSENTRDAQGQFLTVIGAFWHLTKERKYIFSVRMALVNCLKTLLEADPYSKWAILNVMGKDFPVNEVFTQFLADNHHQVRMLAAESINRLFQDTKGDSSRLLKALPLKLQQTAFENAYLKAQEGMREMSHSAENPETLDEIYNRKSVLLTLIAVVLSCSPICEKQALFALCKSVKENGLEPHLVKKVLEKVSETFGYRRLEDFMASHLDYLVLEWLNLQDTEYNLSSFPFILLNYTNIEDFYRSCYKVLIPHLVIRSHFDEVKSIANQIQEDWKSLLTDCFPKILVNILPYFAYEGTRDSGMAQQRETATKVYDMLKSENLLGKQIDHLFISNLPEIVVELLMTLHEPANSSASQSTDLCDFSGDLDPAPNPPHFPSHVIKATFAYISNCHKTKLKSILEILSKSPDSYQKILLAICEQAAETNNVYKKHRILKIYHLFVSLLLKDIKSGLGGAWAFVLRDVIYTLIHYINQRPSCIMDVSLRSFSLCCDLLSQVCQTAVTYCKDALENHLHVIVGTLIPLVYEQVEVQKQVLDLLKYLVIDNKDNENLYITIKLLDPFPDHVVFKDLRITQQKIKYSRGPFSLLEEINHFLSVSVYDALPLTRLEGLKDLRRQLELHKDQMVDIMRASQDNPQDGIMVKLVVNLLQLSKMAINHTGEKEVLEAVGSCLGEVGPIDFSTIAIQHSKDASYTKALKLFEDKELQWTFIMLTYLNNTLVEDCVKVRSAAVTCLKNILATKTGHSFWEIYKMTTDPMLAYLQPFRTSRKKFLEVPRFDKENPFEGLDDINLWIPLSENHDIWIKTLTCAFLDSGGTKCEILQLLKPMCEVKTDFCQTVLPYLIHDILLQDTNESWRNLLSTHVQGFFTSCLRHFSQTSRSTTPANLDSESEHFFRCCLDKKSQRTMLAVVDYMRRQKRPSSGTIFNDAFWLDLNYLEVAKVAQSCAAHFTALLYAEIYADKKSMDDQEKRSLAFEEGSQNTTISSLSEKSKEETGISLQDLLLEIYRSIGEPDSLYGCGGGKMLQPITRLRTYEHEAMWGKALVTYDLETAIPSSTRQAGIIQALQNLGLCHILSVYLKGLDYENKDWCPELEELHYQAAWRNMQWDHCTSVSKEVEGTSYHESLYNALQSLRDREFSTFYESLKYARVKEVEEMCKRSLESVYSLYPTLSRLQAIGELESIGELFSRSVTHRQLSEVYIKWQKHSQLLKDSDFSFQEPIMALRTVILEILMEKEMDNSQRECIKDILTKHLVELSILARTFKNTQLPERAIFQIKQYNSVSCGVSEWQLEEAQVFWAKKEQSLALSILKQMIKKLDASCAANNPSLKLTYTECLRVCGNWLAETCLENPAVIMQTYLEKAVEVAGNYDGESSDELRNGKMKAFLSLARFSDTQYQRIENYMKSSEFENKQALLKRAKEEVGLLREHKIQTNRYTVKVQRELELDELALRALKEDRKRFLCKAVENYINCLLSGEEHDMWVFRLCSLWLENSGVSEVNGMMKANGMKIPTYKFLPLMYQLAARMGTKMMGGLGFHEVLNNLISRISMDHPHHTLFIILALANANRDEFLTKPEVARRSRITKNVPKQSSQLDEDRTEAANRIICTIRSRRPQMVRSVEALCDAYIILANLDATQWKTQRKGINIPADQPITKLKNLEDVVVPTMEIKVDHTGEYGNLVTIQSFKAEFRLAGGVNLPKIIDCVGSDGKERRQLVKGRDDLRQDAVMQQVFQMCNTLLQRNTETRKRKLTICTYKVVPLSQRSGVLEWCTGTVPIGEFLVNNEDGAHKRYRPNDFSAFQCQKKMMEVQKKSFEEKYEVFMDVCQNFQPVFRYFCMEKFLDPAIWFEKRLAYTRSVATSSIVGYILGLGDRHVQNILINEQSAELVHIDLGVAFEQGKILPTPETVPFRLTRDIVDGMGITGVEGVFRRCCEKTMEVMRNSQETLLTIVEVLLYDPLFDWTMNPLKALYLQQRPEDETELHPTLNADDQECKRNLSDIDQSFNKVAERVLMRLQEKLKGVEEGTVLSVGGQVNLLIQQAIDPKNLSRLFPGWKAWV",
    #                    }

    # embed_tensor = get_many_embeddings(kinase_seq_dict, True)
    # kinase_distance_map = calc_embedding_distance_map(embed_tensor, distance_type='euclidean')
    # k_nearest_neighbors = get_k_nearest_neighbors(kinase_distance_map, k=2)
    #
    # print(kinase_distance_map)
    # print("k nearest indices", k_nearest_neighbors)
    #
    # k_nearest_distances = torch.gather(kinase_distance_map, 1, k_nearest_neighbors)
    # print(k_nearest_distances)
    # print(k_nearest_distances.shape)

    # Test the wrapper function
    # negative_pairs = get_negative_kinase_distance_pairs(df, k=5, progress_bar=True)
    # print(negative_pairs)
    # print(negative_pairs.shape)

    name_pairs = get_negative_kinase_name_pairs(df, k=5, progress_bar=True)
    print(name_pairs[0])