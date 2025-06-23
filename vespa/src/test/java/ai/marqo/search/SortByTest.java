package ai.marqo.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import org.junit.jupiter.api.Test;

/**
 * Test for sortBy feature in HybridSearcher.
 */
class SortByTest {

    /**
     * Test that verifies sorting of results based on a single sort field.
     */
    HitGroup helpGenerateHitGroupWithOnlySortFieldValue0() {
        FeatureData f1 = mock(FeatureData.class);
        when(f1.getDouble("sort_field_value_0")).thenReturn(-1e50);
        Hit doc1 = new Hit("doc1", 0.55);
        doc1.setField("matchfeatures", f1);

        FeatureData f2 = mock(FeatureData.class);
        when(f2.getDouble("sort_field_value_0")).thenReturn(1.0);
        Hit doc2 = new Hit("doc2", 0.65);
        doc2.setField("matchfeatures", f2);

        FeatureData f3 = mock(FeatureData.class);
        when(f3.getDouble("sort_field_value_0")).thenReturn(2.0);
        Hit doc3 = new Hit("doc3", 0.75);
        doc3.setField("matchfeatures", f3);

        FeatureData f4 = mock(FeatureData.class);
        when(f4.getDouble("sort_field_value_0")).thenReturn(2.0);
        Hit doc4 = new Hit("doc4", 0.85);
        doc4.setField("matchfeatures", f4);

        FeatureData f5 = mock(FeatureData.class);
        when(f5.getDouble("sort_field_value_0")).thenReturn(-1e50);
        Hit doc5 = new Hit("doc5", 0.45);
        doc5.setField("matchfeatures", f5);

        FeatureData f6 = mock(FeatureData.class);
        when(f6.getDouble("sort_field_value_0")).thenReturn(5.0);
        Hit doc6 = new Hit("doc6", 0.95);
        doc6.setField("matchfeatures", f6);

        FeatureData f7 = mock(FeatureData.class);
        when(f7.getDouble("sort_field_value_0")).thenReturn(6.0);
        Hit doc7 = new Hit("doc7", 0.90);
        doc7.setField("matchfeatures", f7);

        FeatureData f8 = mock(FeatureData.class);
        when(f8.getDouble("sort_field_value_0")).thenReturn(7.0);
        Hit doc8 = new Hit("doc8", 0.80);
        doc8.setField("matchfeatures", f8);

        FeatureData f9 = mock(FeatureData.class);
        when(f9.getDouble("sort_field_value_0")).thenReturn(8.0);
        Hit doc9 = new Hit("doc9", 0.70);
        doc9.setField("matchfeatures", f9);

        FeatureData f10 = mock(FeatureData.class);
        when(f10.getDouble("sort_field_value_0")).thenReturn(9.0);
        Hit doc10 = new Hit("doc10", 1.00);
        doc10.setField("matchfeatures", f10);

        HitGroup hits = new HitGroup();
        hits.add(doc1);
        hits.add(doc2);
        hits.add(doc3);
        hits.add(doc4);
        hits.add(doc5);
        hits.add(doc6);
        hits.add(doc7);
        hits.add(doc8);
        hits.add(doc9);
        hits.add(doc10);
        return hits;
    }

    /**
     * Test that verifies sorting of results based on a single sort field with ascending order
     * and last missing policy.
     */
    @Test
    void sort1FieldWithAscOrderAndLastMissingPolicy() {
        // build 10 distinct docs by hand

        HitGroup hitsToSort = helpGenerateHitGroupWithOnlySortFieldValue0();
        HybridSearcher searcher = new HybridSearcher();
        String sortJson =
                "[{"
                        + "\"field_name\":\"ignored\","
                        + "\"order\":\"asc\","
                        + "\"missing\":\"last\""
                        + "}]";

        // full-depth, no trim
        // expected:
        // 1) doc2 (1.0)
        // 2) doc3 & doc4 both 2.0 → tie by original relevance: doc4(0.85) before doc3(0.75)
        // 3) doc6(5),doc7(6),doc8(7),doc9(8),doc10(9)
        // 4) missing last: doc1,doc5
        HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
        assertThat(out.asList())
                .extracting(hit -> hit.getId().toString())
                .containsExactly(
                        "doc2", "doc4", "doc3", "doc6", "doc7", "doc8", "doc9", "doc10", "doc1",
                        "doc5");
    }

    /**
     * Test that verifies sorting of results based on a single sort field with desc order
     * and first missing policy.
     */
    @Test
    void sort1FieldWithDescOrderAndFirstMissingPolicy() {
        HitGroup hitsToSort = helpGenerateHitGroupWithOnlySortFieldValue0();
        HybridSearcher searcher = new HybridSearcher();
        String sortJson =
                "[{"
                        + "\"field_name\":\"ignored\","
                        + "\"order\":\"desc\","
                        + "\"missing\":\"first\""
                        + "}]";

        HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
        assertThat(out.asList())
                .extracting(hit -> hit.getId().toString())
                .containsExactly(
                        "doc1", // missing first  (–1e50, highest missing rel=0.55)
                        "doc5", // missing second (–1e50, next missing rel=0.45)
                        "doc10", // sort=9.0
                        "doc9", // sort=8.0
                        "doc8", // sort=7.0
                        "doc7", // sort=6.0
                        "doc6", // sort=5.0
                        "doc4", // sort=2.0, tie-break on original rel=0.85 (before doc3)
                        "doc3", // sort=2.0, tie-break rel=0.75
                        "doc2" // sort=1.0
                        );
    }

    /**
     * Helper that builds 6 docs with two independent sort_field values:
     *  - doc1/doc2 share field0=1.0 but doc2.field1=5.0 < doc1.field1=10.0
     *  - doc3/doc4 share field0=2.0 but doc3.field1=3.0 < doc4.field1=7.0
     *  - doc5/doc6 both missing (–1e50 → null)
     */
    HitGroup helpGenerateHitGroupWithTwoSortFieldValues() {
        HitGroup hits = new HitGroup();
        double[][] values = {
            {1.0, 10.0},
            {1.0, 5.0},
            {2.0, 3.0},
            {2.0, 7.0},
            {-1e50, -1e50},
            {-1e50, -1e50}
        };
        double[] relevances = {0.10, 0.20, 0.30, 0.40, 0.05, 0.06};
        for (int i = 0; i < values.length; i++) {
            FeatureData f = mock(FeatureData.class);
            when(f.getDouble("sort_field_value_0")).thenReturn(values[i][0]);
            when(f.getDouble("sort_field_value_1")).thenReturn(values[i][1]);
            Hit h = new Hit("doc" + (i + 1), relevances[i]);
            h.setField("matchfeatures", f);
            hits.add(h);
        }
        return hits;
    }

    @Test
    void sort2FieldsWithAscOrderAndLastMissingPolicy() {
        HitGroup hitsToSort = helpGenerateHitGroupWithTwoSortFieldValues();
        HybridSearcher searcher = new HybridSearcher();
        // first sort_field_value_0 asc, missing last
        // then sort_field_value_1 asc, missing last
        String sortJson =
                "["
                        + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"last\"},"
                        + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"last\"}"
                        + "]";

        HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
        // Expected:
        // 1) doc2 (1.0,5.0) before doc1 (1.0,10.0)
        // 2) doc3 (2.0,3.0) before doc4 (2.0,7.0)
        // 3) missing last: doc6 (rel=0.06) before doc5 (rel=0.05)
        assertThat(out.asList())
                .extracting(hit -> hit.getId().toString())
                .containsExactly(
                        "doc2", "doc1",
                        "doc3", "doc4",
                        "doc6", "doc5");
    }

    /**
     * Helper that builds 4 docs with three sort_field values:
     *  - docA/B/C all have field0=1.0 but differ on field1/field2
     *  - docD missing all three
     */
    HitGroup helpGenerateHitGroupWithThreeSortFieldValues() {
        HitGroup hits = new HitGroup();
        String[] ids = {"docA", "docB", "docC", "docD"};
        double[][] values = {
            {1.0, 1.0, 3.0}, // docA
            {1.0, 1.0, 2.0}, // docB
            {1.0, 2.0, 1.0}, // docC
            {-1e50, -1e50, -1e50} // docD missing all
        };
        double[] relevances = {0.40, 0.50, 0.60, 0.70};
        for (int i = 0; i < ids.length; i++) {
            FeatureData f = mock(FeatureData.class);
            when(f.getDouble("sort_field_value_0")).thenReturn(values[i][0]);
            when(f.getDouble("sort_field_value_1")).thenReturn(values[i][1]);
            when(f.getDouble("sort_field_value_2")).thenReturn(values[i][2]);
            Hit h = new Hit(ids[i], relevances[i]);
            h.setField("matchfeatures", f);
            hits.add(h);
        }
        return hits;
    }

    @Test
    void sort3FieldsWithAscOrderAndFirstMissingPolicy() {
        HitGroup hitsToSort = helpGenerateHitGroupWithThreeSortFieldValues();
        HybridSearcher searcher = new HybridSearcher();
        // all three ascending, missing first
        String sortJson =
                "[{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"first\"},"
                        + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"first\"},"
                        + "{\"field_name\":\"ignored\",\"order\":\"asc\",\"missing\":\"first\"}]";

        HitGroup out = searcher.postProcessBySort(hitsToSort, sortJson, null, 10, 0);
        // Expected:
        // 1) missing first: docD
        // 2) among the rest, field0 ties=1.0 → use field1:
        //      docA/B have field1=1.0 < docC.field1=2.0 → so docA & docB
        //    then break tie on field2: docB(2.0) < docA(3.0)
        // 3) then docC
        assertThat(out.asList())
                .extracting(hit -> hit.getId().toString())
                .containsExactly("docD", "docB", "docA", "docC");
    }

    private Execution makeEmptyExec() {
        Execution exec = mock(Execution.class);
        when(exec.search(any(Query.class)))
                .thenAnswer(
                        invocation -> {
                            Query q = invocation.getArgument(0);
                            return new Result(q, new HitGroup());
                        });
        return exec;
    }

    /**
     * Test that verifies that postProcessBySort is called when sortBy is set in the query.
     */
    @Test
    void whenSortByFields_set_postProcessBySortIsCalled() {
        // 1) Create a Mockito spy on the real HybridSearcher
        HybridSearcher spy = spy(new HybridSearcher());

        // 2) Stub out createSubQuery (both overloads) so we never NPE inside it
        doAnswer(inv -> inv.getArgument(0))
                .when(spy)
                .createSubQuery(any(Query.class), anyString(), anyString(), anyBoolean());
        doAnswer(inv -> inv.getArgument(0))
                .when(spy)
                .createSubQuery(
                        any(Query.class), anyString(), anyString(), anyBoolean(), anyString());

        // 3) Stub extractTensorRankFeature to return:
        //    • null for "mult_weights_global" or "add_weights_global"
        //    • an empty Tensor for everything else (fields_to_rank_*)
        doAnswer(
                        inv -> {
                            String name = inv.getArgument(1);
                            if (name.contains("mult_weights_global")
                                    || name.contains("add_weights_global")) {
                                return null;
                            }
                            // non-null so createSubQuery and co. won't blow up
                            return Tensor.from("tensor<float>()");
                        })
                .when(spy)
                .extractTensorRankFeature(any(Query.class), anyString());

        // 4) Stub postProcessBySort so it just returns an empty HitGroup
        doReturn(new HitGroup())
                .when(spy)
                .postProcessBySort(any(HitGroup.class), anyString(), any(), anyInt(), anyInt());

        // 5) Build a Query that triggers the sortBy branch
        Query q = new Query("?q");
        q.properties().set("hits", 1);
        q.properties().set("offset", 0);
        q.properties().set("marqo__hybrid.retrievalMethod", "lexical");
        q.properties().set("marqo__hybrid.rankingMethod", "lexical");
        q.properties()
                .set(
                        "marqo__hybrid.sortBy.fields",
                        "[{\"field_name\":\"foo\",\"order\":\"asc\",\"missing\":\"last\"}]");
        // MUST set this to avoid the NPE you saw
        q.properties().set("marqo__hybrid.sortBy.sortCandidates", 10);

        // 6) Call search()
        spy.search(q, makeEmptyExec());

        // 7) Verify that only postProcessBySort() ran
        verify(spy, times(1))
                .postProcessBySort(any(HitGroup.class), anyString(), any(), anyInt(), anyInt());
        verify(spy, never())
                .postProcessResults(any(), any(), any(), anyInt(), anyInt(), anyBoolean());
    }

    /*
       Test that verifies that postProcessResults is called when only modifiers exist
       (i.e., no sortBy.fields).
    */
    @Test
    void whenOnlyModifiersExist_postProcessResultsIsCalled() {
        HybridSearcher spy = spy(new HybridSearcher());
        doAnswer(inv -> inv.getArgument(0))
                .when(spy)
                .createSubQuery(any(), anyString(), anyString(), anyBoolean());
        // simulate "has a global mult modifier" but no sortBy
        Tensor dummy = Tensor.from("tensor<float>(d0[1]):[1]");
        doReturn(dummy).when(spy).extractTensorRankFeature(any(), contains("mult_weights_global"));
        doReturn(null).when(spy).extractTensorRankFeature(any(), contains("add_weights_global"));
        doReturn(new HitGroup())
                .when(spy)
                .postProcessResults(any(), any(), any(), anyInt(), anyInt(), anyBoolean());

        Query q = new Query("?q");
        q.properties().set("hits", 1);
        q.properties().set("offset", 0);
        q.properties().set("marqo__hybrid.retrievalMethod", "lexical");
        q.properties().set("marqo__hybrid.rankingMethod", "lexical");
        // no sortBy.fields

        spy.search(q, makeEmptyExec());

        verify(spy, times(1)).postProcessResults(any(), eq(q), any(), eq(1), eq(0), eq(false));
        verify(spy, never()).postProcessBySort(any(), anyString(), any(), anyInt(), anyInt());
    }
}
