import google.auth
from google.cloud import discoveryengine_v1beta

def search_vertex_ai_search(project_id, location, data_store_id, query):
    """
    Vertex AI Searchで検索を実行する。
    Args:
        project_id (str): Google Cloud プロジェクトID
        location (str): データストアのロケーション ('global' or 'us-central1'など)
        data_store_id (str): データストアのID
        query (str): 検索クエリ
    """
    try:
        # 環境変数から認証情報を取得してクライアントを初期化
        # Discovery Engineでは、クライアントの引数でロケーションを指定しません
        client = discoveryengine_v1beta.SearchServiceClient()

        # 検索リクエストのパスを作成
        serving_config_name = client.serving_config_path(
            project=project_id,
            location=location,
            data_store=data_store_id,
            serving_config='default_serving_config'
        )

        # 検索リクエストを作成
        request = discoveryengine_v1beta.SearchRequest(
            serving_config=serving_config_name,
            query=query
        )

        # 検索を実行
        response = client.search(request)

        # 検索結果を表示
        print("検索結果:")
        for result in response.results:
            document = result.document
            print(f"  - タイトル: {document.derived_struct_data['title']}")
            print(f"  - スニペット: {document.derived_struct_data['snippets'][0]['snippet']}")
            print(f"  - URL: {document.derived_struct_data['link']}")
            print("---")
            
    except Exception as e:
        print(f"検索中にエラーが発生しました: {e}")

# 実行
credentials, project_id = google.auth.default()
# ロケーションとデータストアIDを適切に設定
location = "global"  # もしくは 'us-central1' など
data_store_id = "YOUR_DATA_STORE_ID"
query = "検索したい内容"

search_vertex_ai_search(project_id, location, data_store_id, query)
