import pandas as pd
import numpy as np
from flask import Flask, request, url_for, redirect, render_template, jsonify
import json
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules




def import_data_from_db():
    return ""


def import_and_process_data(file):
    df = pd.read_csv(file, header=0,
                     names=['transactions', 'loyalty', 'days_since_last_contact', 'gender', 'person_id',
                            'product_ids', 'One.hot.1', 'One.hot.2', 'One.hot.3', 'One.hot.4'])

    df['gender'] = pd.Categorical(df['gender'])
    df['One.hot.1'] = pd.Categorical(df['One.hot.1'])
    df['One.hot.2'] = pd.Categorical(df['One.hot.2'])
    df['One.hot.3'] = pd.Categorical(df['One.hot.3'])
    df['One.hot.4'] = pd.Categorical(df['One.hot.4'])
    df['processed_product_ids'] = df['product_ids'].apply(lambda x: np.array(json.loads(x)))

    return df


def encode_data(df, person_id, product):
    shopping_list = df[df['person_id'] == person_id]['processed_product_ids'].to_numpy()[0]
    if product in shopping_list:
        return 1
    else:
        return 0


def build_market_basket(df):
    unique_product_list = df['processed_product_ids'].explode().unique().tolist()
    products_str = [str(x) for x in unique_product_list]

    market_basket = pd.DataFrame(columns=products_str, index=df['person_id'])

    for person_id in df['person_id'].tolist():
        for product in unique_product_list:
            market_basket.loc[person_id, str(product)] = encode_data(df, person_id, product)

    return market_basket


def get_association_rules(market_basket, min_support, min_confidence, max_len=2):
    # compute frequent items using the Apriori algorithm
    frequent_itemsets = apriori(market_basket, min_support=min_support, max_len=max_len, use_colnames=True)

    # compute all association rules for frequent_itemsets
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules


def get_rules_look_up(rules):
    rules_look_up = {}

    for idx in range(0, len(rules['antecedents'])):
        if rules.loc[idx, 'antecedents'] in rules_look_up:
            rules_look_up[rules.loc[idx, 'antecedents']] = rules_look_up[rules.loc[idx, 'antecedents']] + list(
                rules.loc[idx, 'consequents'])
        else:
            rules_look_up[rules.loc[idx, 'antecedents']] = list(rules.loc[idx, 'consequents'])

    return rules_look_up


def int_to_frozenset(product_id):
    return frozenset({str(product_id)})


def strList_to_intList(lst):
    new_list = []
    for i in lst:
        new_list.append(int(i))
    return new_list


def get_shoppingList(df, person_id):
    return df[df['person_id'] == person_id]['processed_product_ids'].to_numpy()[0].tolist()


def get_recommendations(person_id, df, rules_lookup):
    products_to_recommend = []
    
    items_bought = get_shoppingList(df, person_id)
    
    for i in items_bought:
        if int_to_frozenset(i) in rules_lookup:
            recommendations = rules_lookup[int_to_frozenset(i)]
            recommendations = strList_to_intList(recommendations)
            products_to_recommend = products_to_recommend + recommendations
    
    final_recommendations = [item for item in products_to_recommend if item not in items_bought]
    
    if len(final_recommendations) == 0:
        final_recommendations = items_bought
        
    return list(set(final_recommendations))


def main(person_id):
    filepath = '../interview_case_study.csv'

    # import data and pre-process it
    df = import_and_process_data(filepath)

    # generate the market basket data frame
    market_basket = build_market_basket(df)

    # perform market basket analysis
    rules = get_association_rules(market_basket, min_support=0.40, min_confidence=0.50)

    # create look-up table for association rules
    rules_lookup = get_rules_look_up(rules)

    results = get_recommendations(person_id, df, rules_lookup)
	
    return results


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("login.html")


@app.route('/predict', methods=['POST'])
def predict():
    input_data = [x for x in request.form.values()]
    results = main(input_data[0])

    return render_template('home.html', products=results)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    results = main(data[0])
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)